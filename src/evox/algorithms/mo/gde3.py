# --------------------------------------------------------------------------------------
# 1. GDE3 algorithm is described in the following papers:
#
# Title: GDE3: the third evolution step of generalized differential evolution
# Link: https://ieeexplore.ieee.org/document/1554717
#
# 2. This code has been inspired by PlatEMO.
# More information about PlatEMO can be found at the following URL:
# GitHub Link: https://github.com/BIMK/PlatEMO
# --------------------------------------------------------------------------------------

import jax
import jax.numpy as jnp

from evox import Algorithm, jit_class, State
from evox.operators import (
    non_dominated_sort,
    crowding_distance,
    crossover,
)


@jit_class
class GDE3(Algorithm):
    """GDE3 algorithm

    link: https://ieeexplore.ieee.org/document/1554717
    """

    def __init__(
        self,
        lb,
        ub,
        n_objs,
        pop_size,
        F=0.49,
        CR=0.97,
    ):
        """
        Parameters for Differential Evolution
        ----------
        F
            The scaling factor
        CR
            The probability of crossover
        """
        self.lb = lb
        self.ub = ub
        self.n_objs = n_objs
        self.dim = lb.shape[0]
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.de = crossover.DifferentialEvolve(F, CR)

    def setup(self, key):
        key, subkey = jax.random.split(key)
        population = (
            jax.random.uniform(subkey, shape=(self.pop_size, self.dim))
            * (self.ub - self.lb)
            + self.lb
        )
        return State(
            population=population,
            fitness=jnp.zeros((self.pop_size, self.n_objs)),
            next_generation=population,
            is_init=True,
            key=key,
        )

    def ask(self, state):
        return jax.lax.cond(state.is_init, self._ask_init, self._ask_normal, state)

    def tell(self, state, fitness):
        return jax.lax.cond(
            state.is_init, self._tell_init, self._tell_normal, state, fitness
        )

    def _ask_init(self, state):
        return state.population, state

    def _ask_normal(self, state):
        keys = jax.random.split(state.key, 3)
        p = jax.random.choice(keys[1], state.population, (3, self.pop_size))
        differentiated = self.de(keys[2], p[0], p[1], p[2])
        next_gen = jnp.clip(differentiated, self.lb, self.ub)
        return next_gen, state.update(next_generation=next_gen, key=keys[0])

    def _tell_init(self, state, fitness):
        state = state.update(fitness=fitness, is_init=False)
        return state

    def _tell_normal(self, state, fitness):
        merged_pop = jnp.concatenate([state.population, state.next_generation], axis=0)
        merged_fit = jnp.concatenate([state.fitness, fitness], axis=0)

        rank = non_dominated_sort(merged_fit)
        order = jnp.argsort(rank)
        worst_rank = rank[order[self.pop_size]]
        mask = rank == worst_rank
        crowding_dis = crowding_distance(merged_fit, mask)

        combined_order = jnp.lexsort((-crowding_dis, rank))[: self.pop_size]
        survivor = merged_pop[combined_order]
        survivor_fitness = merged_fit[combined_order]
        state = state.update(population=survivor, fitness=survivor_fitness)
        return state
