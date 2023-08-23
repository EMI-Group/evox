# --------------------------------------------------------------------------------------
# 1. This code implements the algorithm described in the following paper:
# Title: Discovering Evolution Strategies via Meta-Black-Box Optimization
# Link: https://arxiv.org/abs/2211.11260
#
# 2. This code has been inspired by or utilizes the algorithmic implementation from evosax.
# More information about evosax can be found at the following URL:
# GitHub Link: https://github.com/RobertTLange/evosax
# --------------------------------------------------------------------------------------

import jax
import jax.numpy as jnp
import evox

def get_des_weights(popsize: int, temperature: float = 12.5):
    """Compute discovered recombination weights."""
    ranks = jnp.arange(popsize)
    ranks /= ranks.size - 1
    ranks = ranks - 0.5
    sigout = jax.nn.sigmoid(temperature * ranks)
    weights = jax.nn.softmax(-20 * sigout)
    return weights


@evox.jit_class
class DES(evox.Algorithm):
    def __init__(
        self,
        pop_size: int,
        center_init: jax.Array,
        temperature: float = 12.5,
        sigma_init: float = 0.1,
        mean_decay: float = 0.0,
    ):

        super().__init__()

        self.num_dims = center_init.shape[0]
        self.center_init = center_init
        self.popsize = pop_size
        self.sigma_init = sigma_init
        self.temperature = temperature
        self.lrate_sigma: float = 0.1  # Learning rate for population std
        self.lrate_mean: float = 1.0  # Learning rate for population mean
        self.init_min: float = 0.0
        self.init_max: float = 0.0
        self.clip_min: float = -jnp.finfo(jnp.float32).max
        self.clip_max: float = jnp.finfo(jnp.float32).max

    def setup(self, key):
        weights = get_des_weights(self.popsize, self.temperature)
        return evox.State(
            key=key,
            sigma=self.sigma_init * jnp.ones(self.num_dims),
            center=self.center_init,
            weights=weights.reshape(-1, 1),
        )

    def ask(self, state):
        key, _ = jax.random.split(state.key)
        z = jax.random.normal(state.key, (self.popsize, self.num_dims))
        x = state.center + z * state.sigma.reshape(1, self.num_dims)
        return x, state.update(key=key, x=x)

    def tell(self, state, fitness):
        x = state.x[fitness.argsort()]
        weights = state.weights
        # Weighted updates
        weighted_mean = (weights * x).sum(axis=0)
        weighted_sigma = jnp.sqrt(
            (weights * (x - state.center) ** 2).sum(axis=0) + 1e-06
        )
        center = state.center + self.lrate_mean * (weighted_mean - state.center)
        sigma = state.sigma + self.lrate_sigma * (weighted_sigma - state.sigma)
        return state.update(center=center, sigma=sigma)
