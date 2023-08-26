# --------------------------------------------------------------------------------------
# 1. This code implements algorithms described in the following papers:
#
# Title: Simplify Your Covariance Matrix Adaptation Evolution Strategy (MAES)
# Link: https://www.honda-ri.de/pubs/pdf/3376.pdf
#
# Title: Limited-Memory Matrix Adaptation for Large Scale Black-box Optimization (LMMAES)
# Link: https://arxiv.org/pdf/1705.06693.pdf
#
# 2. This code has been inspired by or utilizes the algorithmic implementation from evosax.
# More information about evosax can be found at the following URL:
# GitHub Link: https://github.com/RobertTLange/evosax
# --------------------------------------------------------------------------------------

import jax
import jax.numpy as jnp

import evox
from evox import State
from .sort_utils import sort_by_key
from .cma_es import CMAES

@evox.jit_class
class MAES(CMAES):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, key):
        # simplify the setup by removing the covariance matrix C and its square root invsqrtC
        pc = jnp.zeros((self.dim,))
        ps = jnp.zeros((self.dim,))
        M = jnp.eye(self.dim)
        # M replaces the B and D matrices in CMA-ES
        return State(
            pc=pc,
            ps=ps,
            M=M,
            count_iter=0,
            mean=self.center_init,
            sigma=self.init_stdev,
            key=key,
            population=jnp.empty((self.pop_size, self.dim)),
        )

    def ask(self, state):
        key, sample_key = jax.random.split(state.key)
        noise = jax.random.normal(sample_key, (self.pop_size, self.dim))
        population = state.mean + state.sigma * noise @ state.M.T
        new_state = state.update(
            population=population, count_iter=state.count_iter + 1, key=key
        )
        return population, new_state

    def tell(self, state, fitness):
        fitness, population = sort_by_key(fitness, state.population)

        mean = self._update_mean(state.mean, population)
        delta_mean = mean - state.mean

        ps = self._update_ps(state.ps, state.M, state.sigma, delta_mean)

        hsig = (
            jnp.linalg.norm(ps) / jnp.sqrt(1 - (1 - self.cs) ** (2 * state.count_iter))
            < (1.4 + 2 / (self.dim + 1)) * self.chiN
        )
        pc = self._update_pc(state.pc, ps, delta_mean, state.sigma, hsig)

        M = self._update_M(state.M, pc, state.sigma, population, state.mean, hsig)
        sigma = self._update_sigma(state.sigma, ps)

        return state.update(mean=mean, ps=ps, pc=pc, M=M, sigma=sigma)

    def _update_M(self, M, pc, sigma, population, old_mean, hsig):
        y = (population[: self.mu] - old_mean) / sigma
        # The update rule for M in MA-ES is different from the C update in CMA-ES
        M_update = (
            (1 - self.c1 - self.cmu) * M
            + self.c1 * (jnp.outer(pc, pc) + (1 - hsig) * self.cc * (2 - self.cc) * M)
            + self.cmu * (y.T * self.weights) @ y
        )
        return M_update

    # The rest of the methods (_update_mean, _update_ps, _update_pc, _update_sigma) remain the same as in CMAES


@evox.jit_class
class LMMAES(MAES):
    def __init__(self, *args, memory_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_size = memory_size or self.weights.size

    def setup(self, key):
        state = super().setup(key)
        y_memory = jnp.zeros((self.memory_size, self.dim))
        return state.update(y_memory=y_memory)

    def _update_M(self, M, pc, hsig, y_memory):
        weights_reshaped = jnp.reshape(self.weights, (-1, 1))
        M_update = (
            (1 - self.c1 - self.cmu) * M
            + self.c1 * (jnp.outer(pc, pc) + (1 - hsig) * self.cc * (2 - self.cc) * M)
            + self.cmu * weights_reshaped.T @ y_memory
        )
        return M_update

    def tell(self, state, fitness):
        fitness, population = sort_by_key(fitness, state.population)
        mean = self._update_mean(state.mean, population)
        delta_mean = mean - state.mean
        ps = self._update_ps(state.ps, state.M, state.sigma, delta_mean)
        hsig = (
            jnp.linalg.norm(ps) / jnp.sqrt(1 - (1 - self.cs) ** (2 * state.count_iter))
            < (1.4 + 2 / (self.dim + 1)) * self.chiN
        )
        pc = self._update_pc(state.pc, ps, delta_mean, state.sigma, hsig)
        sigma = self._update_sigma(state.sigma, ps)
        y_memory = state.y_memory

        y_memory = jnp.roll(y_memory, shift=-self.mu, axis=0)
        y = (population[: self.mu] - mean) / sigma
        y_memory = y_memory.at[-self.mu :].set(y)

        M = self._update_M(state.M, state.pc, hsig, y_memory)
        return state.update(
            mean=mean, ps=ps, pc=pc, M=M, sigma=sigma, y_memory=y_memory
        )
