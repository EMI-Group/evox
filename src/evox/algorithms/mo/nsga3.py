# --------------------------------------------------------------------------------------
# 1. NSGA-III algorithm is described in the following papers:
#
# Title: An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point-Based Nondominated Sorting
# Approach, Part I: Solving Problems With Box Constraints
# Link: https://ieeexplore.ieee.org/document/6600851
#
# 2. This code has been inspired by PlatEMO.
# More information about PlatEMO can be found at the following URL:
# GitHub Link: https://github.com/BIMK/PlatEMO
# --------------------------------------------------------------------------------------

import jax
import jax.numpy as jnp

from evox.operators import (
    non_dominated_sort,
    selection,
    mutation,
    crossover,
    sampling,
)
from evox import Algorithm, jit_class, State


@jit_class
class NSGA3(Algorithm):
    """NSGA-III algorithm

    link: https://ieeexplore.ieee.org/document/6600851
    """

    def __init__(
        self,
        lb,
        ub,
        n_objs,
        pop_size,
        ref=None,
        selection_op=None,
        mutation_op=None,
        crossover_op=None,
    ):
        self.lb = lb
        self.ub = ub
        self.n_objs = n_objs
        self.dim = lb.shape[0]
        self.pop_size = pop_size
        self.ref = ref

        self.selection = selection_op
        self.mutation = mutation_op
        self.crossover = crossover_op

        if self.ref is None:
            self.ref = sampling.UniformSampling(pop_size, n_objs)()[0]
        if self.selection is None:
            self.selection = selection.UniformRand(0.5)
        if self.mutation is None:
            self.mutation = mutation.Gaussian()
        if self.crossover is None:
            self.crossover = crossover.UniformRand()

        self.ref = self.ref / jnp.linalg.norm(self.ref, axis=1)[:, None]

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

    def init_ask(self, state):
        return state.population, state

    def init_tell(self, state, fitness):
        state = state.update(fitness=fitness)
        return state

    def ask(self, state):
        key, sel_key1, mut_key, sel_key2, x_key = jax.random.split(state.key, 5)
        selected = self.selection(sel_key1, state.population)
        mutated = self.mutation(mut_key, selected)
        selected = self.selection(sel_key2, state.population)
        crossovered = self.crossover(x_key, selected)

        next_generation = jnp.clip(
            jnp.concatenate([mutated, crossovered], axis=0), self.lb, self.ub
        )
        return next_generation, state.update(next_generation=next_generation, key=key)

    def tell(self, state, fitness):
        merged_pop = jnp.concatenate([state.population, state.next_generation], axis=0)
        merged_fitness = jnp.concatenate([state.fitness, fitness], axis=0)

        rank = non_dominated_sort(merged_fitness)
        order = jnp.argsort(rank)
        rank = rank[order]
        ranked_pop = merged_pop[order]
        ranked_fitness = merged_fitness[order]
        last_rank = rank[self.pop_size]
        ranked_fitness = jnp.where(
            jnp.repeat((rank <= last_rank)[:, None], self.n_objs, axis=1),
            ranked_fitness,
            jnp.nan,
        )

        # Normalize
        ideal = jnp.nanmin(ranked_fitness, axis=0)
        offset_fitness = ranked_fitness - ideal
        weight = jnp.eye(self.n_objs, self.n_objs) + 1e-6
        weighted = (
            jnp.repeat(offset_fitness, self.n_objs, axis=0).reshape(
                len(offset_fitness), self.n_objs, self.n_objs
            )
            / weight
        )
        asf = jnp.nanmax(weighted, axis=2)
        ex_idx = jnp.argmin(asf, axis=0)
        extreme = offset_fitness[ex_idx]

        def extreme_point(val):
            extreme = val[0]
            plane = jnp.linalg.solve(extreme, jnp.ones(self.n_objs))
            intercept = 1 / plane
            return intercept

        def worst_point(val):
            return jnp.nanmax(ranked_fitness, axis=0)

        nadir_point = jax.lax.cond(
            jnp.linalg.matrix_rank(extreme) == self.n_objs,
            extreme_point,
            worst_point,
            (extreme, offset_fitness),
        )
        normalized_fitness = offset_fitness / nadir_point

        # Associate
        def perpendicular_distance(x, y):
            proj_len = x @ y.T
            proj_vec = proj_len.reshape(proj_len.size, 1) * jnp.tile(y, (len(x), 1))
            prep_vec = jnp.repeat(x, len(y), axis=0) - proj_vec
            dist = jnp.reshape(jnp.linalg.norm(prep_vec, axis=1), (len(x), len(y)))
            return dist

        dist = perpendicular_distance(ranked_fitness, self.ref)
        pi = jnp.nanargmin(dist, axis=1)
        d = dist[jnp.arange(len(normalized_fitness)), pi]

        # Niche
        def niche_loop(val):
            def nope(val):
                idx, i, rho, j = val
                rho = rho.at[j].set(self.pop_size)
                return idx, i, rho, j

            def have(val):
                def zero(val):
                    idx, i, rho, j = val
                    idx = idx.at[i].set(jnp.nanargmin(jnp.where(pi == j, d, jnp.nan)))
                    rho = rho.at[j].add(1)
                    return idx, i + 1, rho, j

                def already(val):
                    idx, i, rho, j = val
                    key = jax.random.PRNGKey(i * j)
                    temp = jax.random.randint(
                        key, (1, len(ranked_pop)), 0, self.pop_size
                    )
                    temp = temp + (pi == j) * self.pop_size
                    idx = idx.at[i].set(jnp.argmax(temp))
                    rho = rho.at[j].add(1)
                    return idx, i + 1, rho, j

                return jax.lax.cond(rho[val[3]], already, zero, val)

            idx, i, rho = val
            j = jnp.argmin(rho)
            idx, i, rho, j = jax.lax.cond(
                jnp.sum(pi == j), have, nope, (idx, i, rho, j)
            )
            return idx, i, rho

        survivor_idx = jnp.arange(self.pop_size)
        rho = jnp.bincount(
            jnp.where(rank < last_rank, pi, len(self.ref)), length=len(self.ref)
        )
        pi = jnp.where(rank == last_rank, pi, -1)
        d = jnp.where(rank == last_rank, d, jnp.nan)
        survivor_idx, _, _ = jax.lax.while_loop(
            lambda val: val[1] < self.pop_size,
            niche_loop,
            (survivor_idx, jnp.sum(rho), rho),
        )

        state = state.update(
            population=ranked_pop[survivor_idx], fitness=ranked_fitness[survivor_idx]
        )
        return state
