# --------------------------------------------------------------------------------------
# 1. This code implements algorithms described in the following papers:
#
# Title: Benchmarking Parameter-Free AMaLGaM on Functions With and Without Noise(AMaLGaM, IndependentAMaLGaM)
# Link: https://homepages.cwi.nl/~bosman/publications/2013_benchmarkingparameterfree.pdf
#
# 2. This code has been inspired by or utilizes the algorithmic implementation from evosax.
# More information about evosax can be found at the following URL:
# GitHub Link: https://github.com/RobertTLange/evosax
# --------------------------------------------------------------------------------------

import jax
import jax.numpy as jnp
from jax import lax

from evox import State
from .cma_es import CMAES
from .sort_utils import sort_by_key


# Full AMaLGaM implementation
class AMaLGaM(CMAES):

    # Override the mean update to include Anticipated Mean Shift (AMS)
    def _update_mean(self, mean, population, sigma, pc):
        updated_mean = super()._update_mean(mean, population)
        alpha = 0.05  # You can adjust this value based on your needs
        return updated_mean + alpha * sigma * pc

    # Override C update to include Adaptive Variance Scaling (AVS)
    def _update_C(self, C, pc, sigma, population, old_mean, hsig):
        updated_C = super()._update_C(C, pc, sigma, population, old_mean, hsig)
        y = (population[: self.mu] - old_mean) / sigma
        sdr = jnp.linalg.norm(y[0]) / sigma
        beta = 0.05
        true_fun = lambda _: (1 + beta * (sdr - 1)) * C
        false_fun = lambda _: (1 / (1 + beta * (1 - sdr))) * C
        return jax.lax.cond(sdr > 1, None, true_fun, None, false_fun)

    def tell(self, state, fitness):
        fitness, population = sort_by_key(fitness, state.population)

        mean = self._update_mean(state.mean, population, state.sigma, state.pc)
        delta_mean = mean - state.mean

        ps = self._update_ps(state.ps, state.invsqrtC, state.sigma, delta_mean)

        hsig = (
            jnp.linalg.norm(ps) / jnp.sqrt(1 - (1 - self.cs) ** (2 * state.count_iter))
            < (1.4 + 2 / (self.dim + 1)) * self.chiN
        )
        pc = self._update_pc(state.pc, ps, delta_mean, state.sigma, hsig)
        C = self._update_C(state.C, pc, state.sigma, population, state.mean, hsig)
        sigma = self._update_sigma(state.sigma, ps)

        B, D, invsqrtC = lax.cond(
            state.count_iter % self.decomp_per_iter == 0,
            self._decomposition_C,
            lambda _C: (state.B, state.D, state.invsqrtC),
            C,
        )

        return state.update(
            mean=mean, ps=ps, pc=pc, C=C, sigma=sigma, B=B, D=D, invsqrtC=invsqrtC
        )


class IndependentAMaLGaM(AMaLGaM):
    def setup(self, key):
        # Create a state similar to the parent class, but with C as a vector
        pc = jnp.zeros((self.dim,))
        ps = jnp.zeros((self.dim,))
        C = jnp.ones((self.dim,))

        return State(
            pc=pc,
            ps=ps,
            C=C,
            count_iter=0,
            mean=self.center_init,
            sigma=self.init_stdev,
            key=key,
        )

    def _update_C(self, C, pc, sigma, population, old_mean, hsig):
        y = (population[: self.mu] - old_mean) / sigma
        sdr = jnp.abs(y[0]) / sigma

        # update C for single dimension
        C = (
            (1 - self.c1 - self.cmu) * C
            + self.c1 * ((pc**2) + (1 - hsig) * self.cc * (2 - self.cc) * C)
            + self.cmu * self.weights @ (y**2)
        )

        beta = 0.05
        updated_C = jnp.where(
            sdr > 1, (1 + beta * (sdr - 1)) * C, (1 / (1 + beta * (1 - sdr))) * C
        )

        return updated_C

    def ask(self, state):
        key, sample_key = jax.random.split(state.key)
        noise = jax.random.normal(sample_key, (self.pop_size, self.dim))

        # Scale the noise for each dimension using C
        population = state.mean + state.sigma * (jnp.sqrt(state.C) * noise)

        new_state = state.update(
            population=population, count_iter=state.count_iter + 1, key=key
        )

        return population, new_state

    def tell(self, state, fitness):
        fitness, population = sort_by_key(fitness, state.population)

        mean = self._update_mean(state.mean, population, state.sigma, state.pc)
        delta_mean = mean - state.mean

        ps = self._update_ps(state.ps, state.C, state.sigma, delta_mean)

        hsig = (
            jnp.linalg.norm(ps) / jnp.sqrt(1 - (1 - self.cs) ** (2 * state.count_iter))
            < (1.4 + 2 / (self.dim + 1)) * self.chiN
        )

        pc = self._update_pc(state.pc, ps, delta_mean, state.sigma, hsig)
        C = self._update_C(state.C, pc, state.sigma, population, state.mean, hsig)
        sigma = self._update_sigma(state.sigma, ps)

        return state.update(mean=mean, ps=ps, pc=pc, C=C, sigma=sigma)
