# --------------------------------------------------------------------------------------
# This code implements algorithms described in the following papers:
#
# Title: Exponential Natural Evolution Strategies (XNES)
# Link: https://dl.acm.org/doi/abs/10.1145/1830483.1830557
#
# Title: Natural Evolution Strategies (SeparableNES)
# Link: https://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf
# --------------------------------------------------------------------------------------

import math
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm
from .sort_utils import sort_by_key
from evox import Algorithm, State, jit_class


@jit_class
class XNES(Algorithm):
    def __init__(
        self,
        init_mean,
        init_covar,
        pop_size=None,
        recombination_weights=None,
        learning_rate_mean=None,
        learning_rate_var=None,
        learning_rate_B=None,
        covar_as_cholesky=False,
    ):
        """
        See [link](https://arxiv.org/pdf/1106.4487.pdf) for default parameters
        """
        self.dim = init_mean.shape[0]
        self.init_mean = init_mean
        if pop_size is None:
            pop_size = 4 + math.floor(3 * math.log(self.dim))
        self.pop_size = pop_size
        assert pop_size > 0

        if learning_rate_mean is None:
            learning_rate_mean = 1
        self.learning_rate_mean = learning_rate_mean
        if learning_rate_var is None:
            learning_rate_var = (
                (9 + 3 * math.log(self.dim)) / 5 / math.pow(self.dim, 1.5)
            )
        self.learning_rate_var = learning_rate_var
        if learning_rate_B is None:
            learning_rate_B = learning_rate_var
        self.learning_rate_B = learning_rate_B
        assert learning_rate_mean > 0 and learning_rate_var > 0 and learning_rate_B > 0
        if not covar_as_cholesky:
            init_covar = jnp.linalg.cholesky(init_covar)
        self.init_covar = init_covar

        if recombination_weights is None:
            recombination_weights = math.log(pop_size / 2 + 1) - jnp.log(
                jnp.arange(1, pop_size + 1)
            )
            recombination_weights = jnp.clip(recombination_weights, 0)
            recombination_weights = (
                recombination_weights / jnp.sum(recombination_weights) - 1 / pop_size
            )

        self.recombination_weights = recombination_weights
        assert (
            self.recombination_weights[1:] <= self.recombination_weights[:-1]
        ).all(), "recombination_weights must be in descending order"

    def setup(self, key):
        mean = self.init_mean
        sigma = math.pow(jnp.prod(jnp.diag(self.init_covar)), 1 / self.dim)
        B = self.init_covar / sigma
        population = jnp.empty((self.pop_size, self.dim))
        noise = jnp.empty_like(population)

        return State(
            noise=noise,
            mean=mean,
            sigma=sigma,
            B=B,
            key=key,
        )

    def ask(self, state):
        key, normal_key = jax.random.split(state.key)
        noise = jax.random.normal(normal_key, shape=(self.pop_size, self.dim))
        population = state.mean + state.sigma * (noise @ state.B.T)
        return population, state.update(noise=noise, key=key)

    def tell(self, state, fitness):
        if self.recombination_weights is None:
            weights = fitness
            noise = state.noise
        else:
            fitness, noise = sort_by_key(fitness, state.noise)
            weights = self.recombination_weights

        I = jnp.eye(self.dim)
        grad_delta = jnp.sum(weights[:, jnp.newaxis] * noise, axis=0)
        grad_M = (weights * noise.T) @ noise - jnp.sum(weights) * I
        grad_sigma = jnp.trace(grad_M) / self.dim
        grad_B = grad_M - grad_sigma * I
        mean = state.mean + self.learning_rate_mean * state.sigma * state.B @ grad_delta
        sigma = state.sigma * jnp.exp(self.learning_rate_var / 2 * grad_sigma)
        B = state.B @ expm(self.learning_rate_B / 2 * grad_B)

        return state.update(
            mean=mean,
            sigma=sigma,
            B=B,
        )


@jit_class
class SeparableNES(Algorithm):
    def __init__(
        self,
        init_mean,
        init_std,
        pop_size=None,
        recombination_weights=None,
        learning_rate_mean=None,
        learning_rate_var=None,
    ):
        """
        See [link](https://arxiv.org/pdf/1106.4487.pdf) for default parameters
        """
        self.dim = init_mean.shape[0]
        self.init_mean = init_mean
        assert init_std.shape == (self.dim,)
        self.init_std = init_std
        if pop_size is None:
            pop_size = 4 + math.floor(3 * math.log(self.dim))
        self.pop_size = pop_size
        assert pop_size > 0

        if learning_rate_mean is None:
            learning_rate_mean = 1
        self.learning_rate_mean = learning_rate_mean
        if learning_rate_var is None:
            learning_rate_var = (3 + math.log(self.dim)) / 5 / math.sqrt(self.dim)
        self.learning_rate_var = learning_rate_var
        assert learning_rate_mean > 0 and learning_rate_var > 0

        if recombination_weights is None:
            recombination_weights = math.log(pop_size / 2 + 1) - jnp.log(
                jnp.arange(1, pop_size + 1)
            )
            recombination_weights = jnp.clip(recombination_weights, 0)
            recombination_weights = (
                recombination_weights / jnp.sum(recombination_weights) - 1 / pop_size
            )
        else:
            assert recombination_weights.shape == (self.dim,)
        self.weight = recombination_weights

    def _new_pop(self, key, mean, sigma):
        key, sample_key = jax.random.split(key)
        zero_mean_pop = jax.random.normal(sample_key, shape=(self.pop_size, self.dim))
        population = jax.vmap(lambda p: mean + sigma * p)(zero_mean_pop)
        return population, zero_mean_pop, key

    def setup(self, key):
        mean = self.init_mean
        sigma = self.init_std

        population, zero_mean_pop, key = self._new_pop(key, mean, sigma)
        return State(
            population=population,
            zero_mean_pop=zero_mean_pop,
            mean=mean,
            sigma=sigma,
            key=key,
        )

    def ask(self, state):
        return state.population, state

    def tell(self, state, fitness):
        mean = state.mean
        sigma = state.sigma
        population = state.population
        zero_mean_pop = state.zero_mean_pop

        fitness, population, zero_mean_pop = sort_key_valrows(
            fitness, population, zero_mean_pop
        )

        grad_μ = jnp.sum(
            jax.vmap(lambda u, s: u * s)(self.weight, zero_mean_pop), axis=0
        )
        grad_sigma = jnp.sum(
            jax.vmap(lambda u, s: u * (s * s - 1))(self.weight, zero_mean_pop), axis=0
        )

        mean += self.learning_rate_mean * sigma * grad_μ
        sigma *= jnp.exp(self.learning_rate_var / 2 * grad_sigma)

        population, zero_mean_pop, key = self._new_pop(state.key, mean, sigma)
        return state.update(
            population=population,
            zero_mean_pop=zero_mean_pop,
            mean=mean,
            sigma=sigma,
            key=key,
        )
