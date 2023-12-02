# --------------------------------------------------------------------------------------
# 1. This code implements the algorithm described in the following paper:
# Title: Discovering Evolution Strategies via Meta-Black-Box Optimization
# Link: https://arxiv.org/abs/2211.11260
#
# 2. This code has been inspired by or utilizes the algorithmic implementation from evosax.
# More information about evosax can be found at the following URL:
# GitHub Link: https://github.com/RobertTLange/evosax
# --------------------------------------------------------------------------------------
import sys
import functools
import pkgutil
import jax
import jax.numpy as jnp
import evox
from flax import linen as nn

from typing import Optional

if sys.version_info < (3, 8):
    import pickle5 as pickle
else:
    import pickle


def load_pkl_object(filename, pkg_load: bool = False):
    """Reload pickle objects from path."""
    if not pkg_load:
        with open(filename, "rb") as input:
            obj = pickle.load(input)
    else:
        obj = pickle.loads(filename)
    return obj


def tanh_timestamp(x: jax.Array) -> jax.Array:
    def single_frequency(timescale):
        return jnp.tanh(x / jnp.float32(timescale) - 1.0)

    all_frequencies = jnp.asarray(
        [1, 3, 10, 30, 50, 100, 250, 500, 750, 1000, 1250, 1500, 2000],
        dtype=jnp.float32,
    )
    return jax.vmap(single_frequency)(all_frequencies)


def compute_ranks(fitness: jax.Array) -> jax.Array:
    """Return fitness ranks in [0, len(fitness))."""
    ranks = jnp.zeros(len(fitness))
    ranks = ranks.at[fitness.argsort()].set(jnp.arange(len(fitness)))
    return ranks


def z_score_trafo(arr: jax.Array) -> jax.Array:
    """Make fitness 'Gaussian' by substracting mean and dividing by std."""
    return (arr - jnp.nanmean(arr)) / (jnp.nanstd(arr) + 1e-10)


def centered_rank_trafo(fitness: jax.Array) -> jax.Array:
    """Return ~ -0.5 to 0.5 centered ranks (best to worst - min!)."""
    y = compute_ranks(fitness)
    y /= fitness.size - 1
    return y - 0.5


def compute_l2_norm(x: jax.Array) -> jax.Array:
    """Compute L2-norm of x_i. Assumes x to have shape (popsize, num_dims)."""
    return jnp.nanmean(x * x, axis=1)


def range_norm_trafo(
    arr: jax.Array, min_val: float = -1.0, max_val: float = 1.0
) -> jax.Array:
    """Map scores into a min/max range."""
    arr = jnp.clip(arr, -1e10, 1e10)
    normalized_arr = (max_val - min_val) * (arr - jnp.nanmin(arr)) / (
        jnp.nanmax(arr) - jnp.nanmin(arr) + 1e-10
    ) + min_val
    return normalized_arr


class EvolutionPath(object):
    def __init__(self, num_dims: int, timescales):
        self.num_dims = num_dims
        self.timescales = timescales

    def initialize(self):
        """Initialize evolution path arrays."""
        return jnp.zeros((self.num_dims, self.timescales.shape[0]))

    def update(self, paths, diff):
        """Batch update evolution paths for multiple dims & timescales."""

        def update_path(lrate, path, diff):
            return (1 - lrate) * path + (1 - lrate) * diff

        return jax.vmap(update_path, in_axes=(0, 1, None), out_axes=1)(
            self.timescales, paths, diff
        )


class FitnessFeatures(object):
    def __init__(
        self,
        centered_rank: bool = False,
        z_score: bool = False,
        w_decay: float = 0.0,
        diff_best: bool = False,
        norm_range: bool = False,
        maximize: bool = False,
    ):
        self.centered_rank = centered_rank
        self.z_score = z_score
        self.w_decay = w_decay
        self.diff_best = diff_best
        self.norm_range = norm_range
        self.maximize = maximize

    @functools.partial(jax.jit, static_argnums=0)
    def apply(self, x: jax.Array, fitness: jax.Array, best_fitness: float) -> jax.Array:
        """Compute and concatenate different fitness transformations."""
        fitness = jax.lax.select(self.maximize, -1 * fitness, fitness)
        fit_out = ((fitness < best_fitness) * 1.0).reshape(-1, 1)

        if self.centered_rank:
            fit_cr = centered_rank_trafo(fitness).reshape(-1, 1)
            fit_out = jnp.concatenate([fit_out, fit_cr], axis=1)
        if self.z_score:
            fit_zs = z_score_trafo(fitness).reshape(-1, 1)
            fit_out = jnp.concatenate([fit_out, fit_zs], axis=1)
        if self.diff_best:
            fit_best = norm_diff_best(fitness, best_fitness).reshape(-1, 1)
            fit_out = jnp.concatenate([fit_out, fit_best], axis=1)
        if self.norm_range:
            fit_norm = range_norm_trafo(fitness, -1.0, 1.0).reshape(-1, 1)
            fit_out = jnp.concatenate([fit_out, fit_norm], axis=1)
        if self.w_decay:
            fit_wnorm = compute_l2_norm(x).reshape(-1, 1)
            fit_out = jnp.concatenate([fit_out, fit_wnorm], axis=1)
        return fit_out


def norm_diff_best(fitness: jax.Array, best_fitness: float) -> jax.Array:
    """Normalizes difference from best previous fitness score."""
    fitness = jnp.clip(fitness, -1e10, 1e10)
    diff_best = fitness - best_fitness
    return jnp.clip(
        diff_best / (jnp.nanmax(diff_best) - jnp.nanmin(diff_best.min) + 1e-10),
        -1,
        1,
    )


class AttentionWeights(nn.Module):
    att_hidden_dims: int = 8

    @nn.compact
    def __call__(self, X):
        keys = nn.Dense(self.att_hidden_dims)(X)
        queries = nn.Dense(self.att_hidden_dims)(X)
        values = nn.Dense(1)(X)
        A = nn.softmax(jnp.matmul(queries, keys.T) / jnp.sqrt(X.shape[0]))
        weights = nn.softmax(jnp.matmul(A, values).squeeze())
        return weights[:, None]


class EvoPathMLP(nn.Module):
    """MLP layer for learning rate modulation based on evopaths."""

    mlp_hidden_dims: int = 8

    @nn.compact
    def __call__(
        self,
        path_c: jax.Array,
        path_sigma: jax.Array,
        time_embed: jax.Array,
    ):
        timestamps = jnp.repeat(
            jnp.expand_dims(time_embed, axis=0), repeats=path_c.shape[0], axis=0
        )
        X = jnp.concatenate([path_c, path_sigma, timestamps], axis=1)
        # Perform MLP hidden state update for each solution dim. in parallel
        hidden = jax.vmap(nn.Dense(self.mlp_hidden_dims), in_axes=(0))(X)
        hidden = nn.relu(hidden)
        lrates_mean = nn.sigmoid(nn.Dense(1)(hidden)).squeeze()
        lrates_sigma = nn.sigmoid(nn.Dense(1)(hidden)).squeeze()
        return lrates_mean, lrates_sigma


@evox.jit_class
class LES(evox.Algorithm):
    def __init__(
        self,
        pop_size: int,
        center_init: jax.Array,
        sigma_init: float = 0.1,
        mean_decay: float = 0.0,
        net_params=None,
        net_ckpt_path: Optional[str] = None,
    ):

        super().__init__()

        self.num_dims = center_init.shape[0]
        self.center_init = center_init
        self.popsize = pop_size
        self.sigma_init = sigma_init
        self.init_min: float = -5.0
        self.init_max: float = 5.0
        self.clip_min: float = -jnp.finfo(jnp.float32).max
        self.clip_max: float = jnp.finfo(jnp.float32).max

        self.evopath = EvolutionPath(
            num_dims=self.num_dims, timescales=jnp.array([0.1, 0.5, 0.9])
        )
        self.weight_layer = AttentionWeights(8)
        self.lrate_layer = EvoPathMLP(8)
        self.fitness_features = FitnessFeatures(centered_rank=True, z_score=True)
        self.sigma_init = sigma_init

        # Set net params provided at instantiation
        if net_params is not None:
            self.les_net_params = net_params

        # Load network weights from checkpoint
        if net_ckpt_path is not None:
            self.les_net_params = load_pkl_object(net_ckpt_path)
            print(f"Loaded LES model from ckpt: {net_ckpt_path}")

        if net_params is None and net_ckpt_path is None:
            ckpt_fname = "2023_03_les_v1.pkl"
            data = pkgutil.get_data(__name__, f"{ckpt_fname}")
            self.les_net_params = load_pkl_object(data, pkg_load=True)
            print(f"Loaded pretrained LES model from ckpt: {ckpt_fname}")

    def setup(self, key):
        init_path_c = self.evopath.initialize()
        init_path_sigma = self.evopath.initialize()
        return evox.State(
            key=key,
            sigma=self.sigma_init * jnp.ones(self.num_dims),
            mean=self.center_init,
            path_c=init_path_c,
            path_sigma=init_path_sigma,
            best_fitness=jnp.finfo(jnp.float32).max,
            best_member=self.center_init,
            gen_counter=0,
        )

    def ask(self, state):
        key, _ = jax.random.split(state.key)
        noise = jax.random.normal(state.key, (self.popsize, self.num_dims))
        x = state.mean + noise * state.sigma.reshape(1, self.num_dims)
        x = jnp.clip(x, self.clip_min, self.clip_max)
        return x, state.update(key=key, x=x, noises=noise)

    def tell(self, state, fitness):
        x = state.x
        fit_re = self.fitness_features.apply(x, fitness, state.best_fitness)
        time_embed = tanh_timestamp(state.gen_counter)
        weights = self.weight_layer.apply(self.les_net_params["recomb_weights"], fit_re)
        weight_diff = (weights * (x - state.mean)).sum(axis=0)
        weight_noise = (weights * (x - state.mean) / state.sigma).sum(axis=0)
        path_c = self.evopath.update(state.path_c, weight_diff)
        path_sigma = self.evopath.update(state.path_sigma, weight_noise)
        lrates_mean, lrates_sigma = self.lrate_layer.apply(
            self.les_net_params["lrate_modulation"],
            path_c,
            path_sigma,
            time_embed,
        )
        weighted_mean = (weights * x).sum(axis=0)
        weighted_sigma = jnp.sqrt((weights * (x - state.mean) ** 2).sum(axis=0) + 1e-10)
        mean_change = lrates_mean * (weighted_mean - state.mean)
        sigma_change = lrates_sigma * (weighted_sigma - state.sigma)
        mean = state.mean + mean_change
        sigma = state.sigma + sigma_change
        mean = jnp.clip(mean, self.clip_min, self.clip_max)
        sigma = jnp.clip(sigma, 0, self.clip_max)
        return state.update(
            mean=mean,
            sigma=sigma,
            path_c=path_c,
            path_sigma=path_sigma,
        )
