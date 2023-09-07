# --------------------------------------------------------------------------------------
# 1. EAG-MOEA/D algorithm is described in the following papers:
#
# Title: An External Archive Guided Multiobjective Evolutionary Algorithm Based on Decomposition for Combinatorial
# Optimization
# Link: https://ieeexplore.ieee.org/abstract/document/6882229
#
# 2. This code has been inspired by PlatEMO.
# More information about PlatEMO can be found at the following URL:
# GitHub Link: https://github.com/BIMK/PlatEMO
# --------------------------------------------------------------------------------------

import jax
import jax.numpy as jnp
from functools import partial

from evox import jit_class, Algorithm, State
from evox.operators import (
    selection,
    mutation,
    crossover,
    non_dominated_sort,
    crowding_distance,
)
from evox.operators.sampling import UniformSampling, LatinHypercubeSampling
from evox.utils import pairwise_euclidean_dist


@partial(jax.jit, static_argnums=[1])
def environmental_selection(fitness, n):
    rank = non_dominated_sort(fitness)
    order = jnp.argsort(rank)
    worst_rank = rank[order[n - 1]]
    mask = rank == worst_rank
    crowding_dis = crowding_distance(fitness, mask)
    combined_indices = jnp.lexsort((-crowding_dis, rank))[:n]

    return combined_indices


@jit_class
class EAGMOEAD(Algorithm):
    """EAG-MOEA/D algorithm

    link: https://ieeexplore.ieee.org/abstract/document/6882229
    """

    def __init__(
        self,
        lb,
        ub,
        n_objs,
        pop_size,
        LGs=8,
        selection_op=None,
        mutation_op=None,
        crossover_op=None,
    ):
        self.lb = lb
        self.ub = ub
        self.n_objs = n_objs
        self.dim = lb.shape[0]
        self.pop_size = pop_size
        self.LGs = LGs
        self.T = jnp.ceil(self.pop_size / 10).astype(int)

        self.selection = selection_op
        self.mutation = mutation_op
        self.crossover = crossover_op

        if self.selection is None:
            self.selection = selection.RouletteWheelSelection(self.pop_size)
        if self.mutation is None:
            self.mutation = mutation.Polynomial((lb, ub))
        if self.crossover is None:
            self.crossover = crossover.SimulatedBinary(type=2)
        self.sample = LatinHypercubeSampling(self.pop_size, self.n_objs)

    def setup(self, key):
        key, subkey1, subkey2 = jax.random.split(key, 3)
        ext_archive = (
            jax.random.uniform(subkey1, shape=(self.pop_size, self.dim))
            * (self.ub - self.lb)
            + self.lb
        )
        fitness = jnp.zeros((self.pop_size, self.n_objs))

        w = self.sample(subkey2)[0]
        B = pairwise_euclidean_dist(w, w)
        B = jnp.argsort(B, axis=1)
        B = B[:, : self.T]
        return State(
            population=ext_archive,
            fitness=fitness,
            inner_pop=ext_archive,
            inner_obj=fitness,
            next_generation=ext_archive,
            weight_vector=w,
            B=B,
            s=jnp.zeros((self.pop_size, self.LGs)),
            parent=jnp.zeros((self.pop_size, self.T)).astype(int),
            offspring_loc=jnp.zeros((self.pop_size,)).astype(int),
            gen=0,
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
        key, per_key, sel_key, x_key, mut_key = jax.random.split(state.key, 5)
        B = state.B
        population = state.inner_pop
        n, t = jnp.shape(B)
        s = jnp.sum(state.s, axis=1) + 1e-6
        d = s / jnp.sum(s) + 0.002
        d = d / jnp.sum(d)

        _, offspring_loc = self.selection(sel_key, population, 1.0 / d)
        parent = jnp.zeros((n, 2)).astype(int)
        B = jax.random.permutation(per_key, B, axis=1, independent=True).astype(int)

        def body_fun(i, val):
            val = val.at[i, 0].set(B[offspring_loc[i], 0])
            val = val.at[i, 1].set(B[offspring_loc[i], 1])
            return val.astype(int)

        parent = jax.lax.fori_loop(0, n, body_fun, parent)

        selected_p = jnp.r_[population[parent[:, 0]], population[parent[:, 1]]]

        crossovered = self.crossover(x_key, selected_p)
        next_generation = self.mutation(mut_key, crossovered)

        return next_generation, state.update(
            next_generation=next_generation, offspring_loc=offspring_loc, key=key
        )

    def _tell_init(self, state, fitness):
        state = state.update(fitness=fitness, inner_obj=fitness, is_init=False)
        return state

    def _tell_normal(self, state, fitness):
        gen = state.gen + 1
        ext_archive = state.population
        ext_obj = state.fitness
        inner_pop = state.inner_pop
        inner_obj = state.inner_obj

        offspring = state.next_generation
        offspring_obj = fitness
        B = state.B
        w = state.weight_vector
        s = state.s

        offspring_loc = state.offspring_loc
        vals = (inner_pop, inner_obj)

        def body_fun(i, vals):
            population, pop_obj = vals
            g_old = jnp.sum(
                pop_obj[B[offspring_loc[i], :]] * w[B[offspring_loc[i], :]], axis=1
            )
            g_new = w[B[offspring_loc[i], :]] @ jnp.transpose(offspring_obj[i])
            idx = B[offspring_loc[i]]
            g_new = g_new[:, jnp.newaxis]
            g_old = g_old[:, jnp.newaxis]
            population = population.at[idx].set(
                jnp.where(g_old >= g_new, offspring[i], population[idx])
            )
            pop_obj = pop_obj.at[idx].set(
                jnp.where(g_old >= g_new, offspring_obj[i], pop_obj[idx])
            )
            return (population, pop_obj)

        inner_pop, inner_obj = jax.lax.fori_loop(0, self.pop_size, body_fun, vals)

        merged_pop = jnp.concatenate([ext_archive, offspring], axis=0)
        merged_fitness = jnp.concatenate([ext_obj, offspring_obj], axis=0)

        combined_order = environmental_selection(merged_fitness, self.pop_size)
        survivor = merged_pop[combined_order]
        survivor_fitness = merged_fitness[combined_order]
        mask = combined_order >= self.pop_size
        num_valid = jnp.sum(mask)
        sucessful = jnp.where(mask, size=self.pop_size)

        def update_s(s):
            h = offspring_loc[combined_order[sucessful] - self.pop_size]
            head = h[0]
            h = jnp.where(h == head, -1, h)
            h = h.at[0].set(head)
            hist, _ = jnp.histogram(h, self.pop_size, range=(0, self.pop_size))
            s = s.at[:, gen % self.LGs + 1].set(hist)
            return s

        def no_update(s):
            return s

        s = jax.lax.cond(num_valid != 0, update_s, no_update, s)
        state = state.update(
            population=survivor,
            fitness=survivor_fitness,
            inner_pop=inner_pop,
            inner_obj=inner_obj,
            s=s,
            gen=gen,
        )
        return state
