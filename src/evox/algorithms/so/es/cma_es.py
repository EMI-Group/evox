# --------------------------------------------------------------------------------------
# 1. This code implements algorithms described in the following papers:
#
# Title: Completely Derandomized Self-Adaptation in Evolution Strategies
# Link: http://www.cmap.polytechnique.fr/~nikolaus.hansen/cmaartic.pdf
#
# Title: A Simple Modification in CMA-ES Achieving Linear Time and Space Complexity (SepCMAES)
# Link: https://inria.hal.science/inria-00287367/document
#
# Title: A Restart CMA Evolution Strategy With Increasing Population Size (IPOPCMAES)
# Link: http://www.cmap.polytechnique.fr/~nikolaus.hansen/cec2005ipopcmaes.pdf
#
# Title: Benchmarking a BI-Population CMA-ES on the BBOB-2009 Function Testbed (BIPOPCMAES)
# Link: https://inria.hal.science/inria-00382093/document
#
# 2. This code has been inspired by or utilizes the algorithmic implementation from evosax.
# More information about evosax can be found at the following URL:
# GitHub Link: https://github.com/RobertTLange/evosax
# --------------------------------------------------------------------------------------

import math

import jax
import jax.numpy as jnp
from jax import lax

import evox
from evox import Algorithm, State
from .sort_utils import sort_by_key


@evox.jit_class
class CMAES(Algorithm):
    def __init__(
        self, center_init, init_stdev, pop_size=None, recombination_weights=None, cm=1
    ):
        """
        This implementation follows `The CMA Evolution Strategy: A Tutorial <https://arxiv.org/pdf/1604.00772.pdf>`_.

        .. note::
            CMA-ES involves eigendecomposition,
            which introduces relatively large numerical error,
            and may lead to non-deterministic behavior on different hardware backends.
        """
        self.center_init = center_init
        assert init_stdev > 0, "Expect variance to be a non-negative float"
        self.init_stdev = init_stdev
        self.dim = center_init.shape[0]
        self.cm = cm
        if pop_size is None:
            # auto
            self.pop_size = 4 + math.floor(3 * math.log(self.dim))
        else:
            self.pop_size = pop_size

        if recombination_weights is None:
            # auto
            self.mu = self.pop_size // 2
            self.weights = jnp.log(self.mu + 0.5) - jnp.log(jnp.arange(1, self.mu + 1))
            self.weights = self.weights / sum(self.weights)
        else:
            assert (
                recombination_weights[1:] <= recombination_weights[:-1]
            ).all(), "recombination_weights must be non-increasing"
            assert (
                jnp.abs(jnp.sum(recombination_weights) - 1) < 1e-6
            ), "sum of recombination_weights must be 1"
            assert (
                recombination_weights > 0
            ).all(), "recombination_weights must be positive"
            self.mu = recombination_weights.shape[0]
            assert self.mu <= self.pop_size
            self.weights = recombination_weights

        self.mueff = jnp.sum(self.weights) ** 2 / jnp.sum(self.weights**2)
        # time constant for cumulation for C
        self.cc = (4 + self.mueff / self.dim) / (
            self.dim + 4 + 2 * self.mueff / self.dim
        )

        # t-const for cumulation for sigma control
        self.cs = (2 + self.mueff) / (self.dim + self.mueff + 5)

        # learning rate for rank-one update of C
        self.c1 = 2 / ((self.dim + 1.3) ** 2 + self.mueff)

        # learning rate for rank-μ update of C
        # convert self.dim to float first to prevent overflow
        self.cmu = min(
            1 - self.c1,
            (
                2
                * (self.mueff - 2 + 1 / self.mueff)
                / ((float(self.dim) + 2) ** 2 + self.mueff)
            ),
        )

        # damping for sigma
        self.damps = (
            1 + 2 * max(0, math.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.cs
        )

        self.chiN = self.dim**0.5 * (
            1 - 1 / (4 * self.dim) + 1 / (21 * self.dim**2)
        )
        self.decomp_per_iter = 1 / (self.c1 + self.cmu) / self.dim / 10
        self.decomp_per_iter = max(jnp.floor(self.decomp_per_iter).astype(jnp.int32), 1)

    def setup(self, key):
        pc = jnp.zeros((self.dim,))
        ps = jnp.zeros((self.dim,))
        B = jnp.eye(self.dim)
        D = jnp.ones((self.dim,))
        C = B @ jnp.diag(D) @ B.T
        return State(
            pc=pc,
            ps=ps,
            B=B,
            D=D,
            C=C,
            count_eigen=0,
            count_iter=0,
            invsqrtC=C,
            mean=self.center_init,
            sigma=self.init_stdev,
            key=key,
            population=jnp.empty((self.pop_size, self.dim)),
        )

    def ask(self, state):
        key, sample_key = jax.random.split(state.key)
        noise = jax.random.normal(sample_key, (self.pop_size, self.dim))
        population = state.mean + state.sigma * (state.D * noise) @ state.B.T
        new_state = state.update(
            population=population, count_iter=state.count_iter + 1, key=key
        )
        return population, new_state

    def tell(self, state, fitness):
        fitness, population = sort_by_key(fitness, state.population)

        mean = self._update_mean(state.mean, population)
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

    def _update_mean(self, mean, population):
        update = self.weights @ (population[: self.mu] - mean)
        return mean + self.cm * update

    def _update_ps(self, ps, invsqrtC, sigma, delta_mean):
        return (1 - self.cs) * ps + jnp.sqrt(
            self.cs * (2 - self.cs) * self.mueff
        ) * invsqrtC @ delta_mean / sigma

    def _update_pc(self, pc, ps, delta_mean, sigma, hsig):
        return (1 - self.cc) * pc + hsig * jnp.sqrt(
            self.cc * (2 - self.cc) * self.mueff
        ) * delta_mean / sigma

    def _update_C(self, C, pc, sigma, population, old_mean, hsig):
        y = (population[: self.mu] - old_mean) / sigma
        return (
            (1 - self.c1 - self.cmu) * C
            + self.c1 * (jnp.outer(pc, pc) + (1 - hsig) * self.cc * (2 - self.cc) * C)
            + self.cmu * (y.T * self.weights) @ y
        )

    def _update_sigma(self, sigma, ps):
        return sigma * jnp.exp(
            (self.cs / self.damps) * (jnp.linalg.norm(ps) / self.chiN - 1)
        )

    def _decomposition_C(self, C):
        C = jnp.triu(C) + jnp.triu(C, 1).T  # enforce symmetry
        D, B = jnp.linalg.eigh(C)
        D = jnp.sqrt(D)
        invsqrtC = (B / D) @ B.T
        return B, D, invsqrtC


@evox.jit_class
class SepCMAES(CMAES):
    def setup(self, key):
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

    def ask(self, state):
        key, sample_key = jax.random.split(state.key)
        noise = jax.random.normal(sample_key, (self.pop_size, self.dim))
        population = state.mean + state.sigma * jnp.sqrt(state.C) * noise
        new_state = state.update(
            population=population, count_iter=state.count_iter + 1, key=key
        )
        return population, new_state

    def tell(self, state, fitness):
        fitness, population = sort_by_key(fitness, state.population)

        mean = self._update_mean(state.mean, population)
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

    def _update_ps(self, ps, C, sigma, delta_mean):
        return (1 - self.cs) * ps + jnp.sqrt(
            self.cs * (2 - self.cs) * self.mueff
        ) * delta_mean / jnp.sqrt(C) / sigma

    def _update_C(self, C, pc, sigma, population, old_mean, hsig):
        y = (population[: self.mu] - old_mean) / sigma
        return (
            (1 - self.c1 - self.cmu) * C
            + self.c1 * ((pc**2) + (1 - hsig) * self.cc * (2 - self.cc) * C)
            + self.cmu * self.weights @ (y**2)
        )


@evox.jit_class
class IPOPCMAES(CMAES):
    def __init__(
        self,
        center_init,
        init_stdev,
        pop_size=None,
        recombination_weights=None,
        cm=1,
        stagnation_threshold=50,
    ):
        super().__init__(center_init, init_stdev, pop_size, recombination_weights, cm)

        self.original_pop_size = self.pop_size
        self.restarts = 0
        self.stagnation_threshold = stagnation_threshold
        self.best_fitness = float("inf")
        self.stagnation_count = 0

    def setup(self, key):
        pc = jnp.zeros((self.dim,))
        ps = jnp.zeros((self.dim,))
        B = jnp.eye(self.dim)
        D = jnp.ones((self.dim,))
        C = B @ jnp.diag(D) @ B.T

        return State(
            pc=pc,
            ps=ps,
            B=B,
            D=D,
            C=C,
            count_eigen=0,
            count_iter=0,
            invsqrtC=C,
            mean=self.center_init,
            sigma=self.init_stdev,
            key=key,
            population=jnp.empty((self.pop_size, self.dim)),
            best_fitness=float("inf"),
            restarts=0,
            stagnation_count=0,
            pop_size=self.original_pop_size,
        )

    def _update_best_fitness(self, current_best_fitness, best_fitness):
        return lax.cond(
            current_best_fitness < best_fitness,
            current_best_fitness,
            lambda _: current_best_fitness,
            best_fitness,
            lambda _: best_fitness,
        )

    def tell(self, state, fitness):
        state = super().tell(state, fitness)

        # Update based on the best fitness in the current population
        current_best_fitness = jnp.min(fitness)

        def improve_fn():
            return state.update(best_fitness=current_best_fitness, stagnation_count=0)

        def no_improve_fn():
            new_stagnation_count = self.stagnation_count + 1
            return state.update(
                best_fitness=state.best_fitness, stagnation_count=new_stagnation_count
            )

        new_state = lax.cond(
            current_best_fitness < state.best_fitness, improve_fn, no_improve_fn
        )

        # Check for stagnation
        return lax.cond(
            new_state.stagnation_count >= self.stagnation_threshold,
            new_state,
            self._restart,
            new_state,
            self._remain,
        )

    def _restart(self, state):
        new_restarts = self.restarts + 1
        new_pop_size = self.original_pop_size * (2**new_restarts)
        return state.update(
            restarts=new_restarts,
            pop_size=new_pop_size,
            sigma=self.init_stdev,
            stagnation_count=0,
        )

    def _remain(self, state):
        return state.update(
            restarts=state.restarts,
            pop_size=state.pop_size,
            sigma=self.init_stdev,
            stagnation_count=state.stagnation_count,
        )


class BIPOPCMAES(IPOPCMAES):
    def _restart(self, state):
        # Determine new restarts and population size based on the current population size
        def back_to_original_fn(_):
            new_restarts = self.restarts + 1
            new_pop_size = self.original_pop_size
            return state.update(
                restarts=new_restarts,
                pop_size=new_pop_size,
                sigma=self.init_stdev,
                stagnation_count=0,
            )

        def double_population_fn(_):
            new_restarts = self.restarts + 1
            new_pop_size = self.original_pop_size * (2**new_restarts)
            return state.update(
                restarts=new_restarts,
                pop_size=new_pop_size,
                sigma=self.init_stdev,
                stagnation_count=0,
            )

        # If the current population size is greater than 16 times the original size, restart to original size.
        # Otherwise, double the current population size.
        return jax.lax.cond(
            state.pop_size > 16 * self.original_pop_size,
            None,
            back_to_original_fn,
            None,
            double_population_fn,
        )
