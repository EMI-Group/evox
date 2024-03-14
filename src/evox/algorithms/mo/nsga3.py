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
            self.selection = selection.UniformRand(0.5)
        if self.mutation is None:
            self.mutation = mutation.Gaussian()
        if self.crossover is None:
            self.crossover = crossover.UniformRand()

        self.sampling = sampling.UniformSampling(self.pop_size, self.n_objs)

    def setup(self, key):
        key, subkey = jax.random.split(key)
        population = (
            jax.random.uniform(subkey, shape=(self.pop_size, self.dim))
            * (self.ub - self.lb)
            + self.lb
        )
        self.ref = self.sampling(subkey)[0]
        self.ref = self.ref / jnp.linalg.norm(self.ref, axis=1)[:, None]
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
        key, mut_key, x_key = jax.random.split(state.key, 3)
        crossovered = self.crossover(x_key, state.population)
        next_generation = self.mutation(mut_key, crossovered)
        next_generation = jnp.clip(next_generation, self.lb, self.ub)
        
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
        mark = rank <= last_rank
        ranked_fitness = jnp.where(
           mark[:, None], ranked_fitness, jnp.nan,
        )

        # Normalize
        ideal_points = jnp.nanmin(ranked_fitness, axis=0)
        offset_fitness = ranked_fitness - ideal_points
        weight = jnp.eye(self.n_objs) + 1e-6

        def get_extreme(i):
            return jnp.nanargmin(jnp.nanmax(offset_fitness / weight[i], axis=1))

        extreme_ind = jax.vmap(get_extreme)(jnp.arange(self.n_objs))
        extreme = offset_fitness[extreme_ind]

        # jax.debug.print("extreme:{}", extreme)

        def get_intercept(val):
            # Calculate the intercepts of the hyperplane constructed by the extreme points
            _extreme = val[0]
            plane = jnp.linalg.solve(_extreme, jnp.ones(self.n_objs))
            intercept = 1 / plane
            return intercept

        def worst_intercept(val):
            _ranked_fitness = val[1]
            return jnp.nanmax(_ranked_fitness, axis=0)

        nadir_point = jax.lax.cond(
            jnp.linalg.matrix_rank(extreme) == self.n_objs,
            get_intercept,
            worst_intercept,
            (extreme, ranked_fitness)
        )
        normalized_fitness = ranked_fitness / nadir_point
        # jax.debug.print("normalized_fitness:{}", normalized_fitness)
        def cosin_distance(x, y):
            # normalize
            y = y / jnp.sqrt(jnp.sum(y**2, axis=1, keepdims=True))
            x = x / jnp.sqrt(jnp.nansum(x**2, axis=1, keepdims=True))

            cos_sim = x @ y.T
            cos_dis = 1 - cos_sim
            distance = jnp.sqrt(jnp.nansum(x ** 2, axis=1, keepdims=True)) * jnp.sqrt(1 - cos_dis ** 2)
            # jax.debug.print("cons shape:{}",jnp.sqrt(jnp.nansum(x ** 2, axis=1, keepdims=True)).shape)
            # jax.debug.print("cos_dis shape:{}",jnp.sqrt(1 - cos_dis ** 2).shape)
            return distance

        # dist = perpendicular_distance(ranked_fitness, self.ref)
        dist = cosin_distance(normalized_fitness, self.ref)

        # Associate each solution with its nearest reference point
        group_id = jnp.nanargmin(dist, axis=1)
        group_dist = jnp.nanmin(dist, axis=1)

        rho = jnp.bincount(
            jnp.where(rank == last_rank, group_id, len(self.ref)), length=len(self.ref)
        )
        rho = jnp.where(rho == 0, jnp.nan, rho)
        group_id = jnp.where(rank == last_rank, group_id, -1)
        group_dist = jnp.where(rank == last_rank, group_dist, jnp.nan)
        sleceted_number = jnp.sum(mark)
        # Niche
        def select_loop(val):
            # def nope(val):
            #     idx, i, rho, j = val
            #     rho = rho.at[j].set(self.pop_size)
            #     return idx, i, rho, j
            #
            # def have(val):
            #     def zero(val):
            #         idx, i, rho, j = val
            #         idx = idx.at[i].set(jnp.nanargmin(jnp.where(group_id == j, group_dist, jnp.nan)))
            #         rho = rho.at[j].add(1)
            #         return idx, i + 1, rho, j
            #
            #     def already(val):
            #         idx, i, rho, j = val
            #         key = jax.random.PRNGKey(i * j)
            #         temp = jax.random.randint(
            #             key, (1, len(ranked_pop)), 0, self.pop_size
            #         )
            #         temp = temp + (group_id == j) * self.pop_size
            #         idx = idx.at[i].set(jnp.argmax(temp))
            #         rho = rho.at[j].add(1)
            #         return idx, i + 1, rho, j
            #
            #     return jax.lax.cond(rho[val[3]], already, zero, val)

            rho, rank = val
            group = jnp.nanargmin(rho)
            candidates = jnp.where(group_id == group & rank == last_rank, group_dist, jnp.nan)
            candidate = jnp.nanargmin(candidates)
            rank[candidate] -= 1
            rho[group] += 1

            # idx, i, rho, j = jax.lax.cond(
            #     jnp.sum(group_id == j), have, nope, (idx, i, rho, j)
            # )
            return rho, rank

        # survivor_idx = jnp.arange(self.pop_size)

        # jax.debug.print("len(self.ref):{}", len(self.ref))
        # jax.debug.print("rho:{}", rho)

        rho, rank = jax.lax.while_loop(
            lambda val: jnp.nansum(val[0]) < self.pop_size - sleceted_number,
            select_loop,
            (rho, rank)
        )

        selected_idx = jnp.where(rank<last_rank, jnp.arange(merged_pop.shape[0]), jnp.nan)
        state = state.update(
            population=ranked_pop[selected_idx], fitness=ranked_fitness[selected_idx]
        )
        return state
