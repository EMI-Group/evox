# --------------------------------------------------------------------------------------
# 1. NSGA-II algorithm is described in the following papers:
#
# Title: A fast and elitist multiobjective genetic algorithm: NSGA-II
# Link: https://ieeexplore.ieee.org/document/996017
# --------------------------------------------------------------------------------------

import jax
import jax.numpy as jnp

from evox.operators import (
    non_dominated_sort,
    crowding_distance,
    selection,
    mutation,
    crossover,
)
from evox import Algorithm, jit_class, State


@jit_class
class NSGA2(Algorithm):
    """NSGA-II algorithm

    link: https://ieeexplore.ieee.org/document/996017
    """

    def __init__(
        self,
        lb,
        ub,
        n_objs,
        pop_size,
        selection_op=None,
        mutation_op=None,
        crossover_op=None,
    ):
        self.lb = lb
        self.ub = ub
        self.n_objs = n_objs
        self.dim = lb.shape[0]
        self.pop_size = pop_size

        self.selection = selection_op
        self.mutation = mutation_op
        self.crossover = crossover_op

        if self.selection is None:
            self.selection = selection.UniformRand(1)
        if self.mutation is None:
            self.mutation = mutation.Polynomial((self.lb, self.ub))
        if self.crossover is None:
            self.crossover = crossover.SimulatedBinary()

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
            key=key,
        )

    def init_ask(self, state):
        return state.population, state

    def init_tell(self, state, fitness):
        state = state.update(fitness=fitness)
        return state

    def ask(self, state):
        key, sel_key1, x_key, mut_key = jax.random.split(state.key, 4)
        crossovered = self.selection(sel_key1, state.population, state.fitness)
        crossovered = self.crossover(x_key, state.population)
        next_generation = self.mutation(mut_key, crossovered)

        next_generation = jnp.clip(next_generation, self.lb, self.ub)

        return next_generation, state.update(next_generation=next_generation, key=key)

    def tell(self, state, fitness):
        merged_pop = jnp.concatenate([state.population, state.next_generation], axis=0)
        merged_fitness = jnp.concatenate([state.fitness, fitness], axis=0)

        rank = non_dominated_sort(merged_fitness)
        order = jnp.argsort(rank)
        worst_rank = rank[order[self.pop_size]]
        mask = rank == worst_rank
        crowding_dis = crowding_distance(merged_fitness, mask)

        combined_order = jnp.lexsort((-crowding_dis, rank))[: self.pop_size]
        survivor = merged_pop[combined_order]
        survivor_fitness = merged_fitness[combined_order]
        state = state.update(population=survivor, fitness=survivor_fitness)
        return state
