# --------------------------------------------------------------------------------------
# 1. KnEA algorithm is described in the following papers:
#
# Title: A Knee Point-Driven Evolutionary Algorithm for Many-Objective Optimization
# Link: https://ieeexplore.ieee.org/document/6975108
#
# 2. This code has been inspired by PlatEMO.
# More information about PlatEMO can be found at the following URL:
# GitHub Link: https://github.com/BIMK/PlatEMO
# --------------------------------------------------------------------------------------

import jax
import jax.numpy as jnp
from functools import partial

from evox.operators import (
    selection,
    mutation,
    crossover,
    non_dominated_sort,
)
from evox import Algorithm, jit_class, State
from evox.utils import pairwise_euclidean_dist


@partial(jax.jit, static_argnums=1)
def calc_DW(fit, k):
    dis = pairwise_euclidean_dist(fit, fit)
    order = jnp.argsort(dis, axis=1)
    neighbor = jnp.take_along_axis(dis, order[:, 1 : k + 1], axis=1)
    avg = jnp.sum(neighbor, axis=1) / k
    r = 1 / abs(neighbor - avg[:, None])
    w = r / jnp.sum(r, axis=1)[:, None]
    DW = jnp.sum(neighbor * w, axis=1)
    return DW


@jit_class
class KnEA(Algorithm):
    """KnEA algorithm

    link: https://ieeexplore.ieee.org/document/6975108
    """

    def __init__(
        self,
        lb,
        ub,
        n_objs,
        pop_size,
        knee_rate=0.5,
        k_neighbors=3,
        mutation_op=None,
        crossover_op=None,
    ):
        self.lb = lb
        self.ub = ub
        self.n_objs = n_objs
        self.dim = lb.shape[0]
        self.pop_size = pop_size
        self.knee_rate = knee_rate
        self.k_neighbors = k_neighbors

        self.selection = selection.Tournament(pop_size)
        self.mutation = mutation_op
        self.crossover = crossover_op

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
            knee=jnp.full(self.pop_size, False),
            r=1.0,
            t=0.0,
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
        rank = non_dominated_sort(state.fitness)
        DW = calc_DW(state.fitness, self.k_neighbors)

        keys = jax.random.split(state.key, 4)
        selected, _ = self.selection(keys[1], state.population, -DW, ~state.knee, rank)
        crossovered = self.crossover(keys[2], selected)
        mutated = self.mutation(keys[3], crossovered)
        next_gen = jnp.clip(mutated, self.lb, self.ub)
        return next_gen, state.update(next_generation=next_gen, key=keys[0])

    def _tell_init(self, state, fitness):
        state = state.update(fitness=fitness, is_init=False)
        return state

    def _tell_normal(self, state, fitness):
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

        # find knee point
        def find_knee_loop(i, val):
            def extreme_plane(points):
                return jnp.linalg.solve(points, jnp.ones(self.n_objs))

            def worst_plane(points):
                A = jnp.diag(points.diagonal().clip(min=1e-6))
                return jnp.linalg.solve(A, jnp.ones(self.n_objs))

            knee, r, t, plane, rate, fitness, rank = val
            mask = rank == i
            f_i = jnp.where(mask[:, None].repeat(self.n_objs, axis=1), fitness, jnp.nan)
            mx = jnp.nanmax(f_i, axis=0)
            mn = jnp.nanmin(f_i, axis=0)
            extreme = f_i[jnp.nanargmax(f_i, axis=0)]
            plane = jax.lax.cond(
                jnp.linalg.matrix_rank(extreme) == self.n_objs,
                extreme_plane,
                worst_plane,
                extreme,
            )
            order = jnp.argsort(plane @ f_i.T)  # ordered by dis from L
            r = r * jnp.exp(-(1 - t / rate) / self.n_objs)
            R = (mx - mn) * r

            def select_knee_loop(j, val):
                def is_knee(info):
                    knee, f_i, p, R = info
                    dif = abs(f_i - f_i[p])
                    neighbor = jnp.all(dif < R, axis=1).at[p].set(False)
                    knee = knee & ~neighbor
                    return knee, f_i, p, R

                knee, order, f_i, mask, R = val
                p = order[j]
                info = (knee, f_i, p, R)
                knee, _, _, _ = jax.lax.cond(knee[p], is_knee, lambda x: x, info)
                return knee, order, f_i, mask, R

            init_val = (knee, order, f_i, mask, R)
            knee, _, _, _, _ = jax.lax.fori_loop(
                0, jnp.sum(rank <= i) - jnp.sum(rank < i), select_knee_loop, init_val
            )
            t = jnp.sum((rank <= i) & knee) / jnp.sum(mask)
            return knee, r, t, plane, rate, fitness, rank

        init_val = (
            jnp.full(len(merged_fitness), True),
            state.r,
            state.t,
            jnp.full(self.n_objs, jnp.nan),
            self.knee_rate,
            ranked_fitness,
            rank,
        )
        res = jax.lax.fori_loop(0, last_rank + 1, find_knee_loop, init_val)
        knee, r, t, last_plane, _, _, _ = res
        knee = jnp.where(rank <= last_rank, knee, False)

        # environmental select
        def too_many(info):
            selected, knee, fitness, mask, plane, dif = info
            mask = knee & mask
            fitness = jnp.where(
                jnp.repeat(mask[:, None], self.n_objs, axis=1), fitness, jnp.nan
            )
            order = jnp.argsort(-plane @ fitness.T)
            idx = jnp.where(jnp.arange(len(order)) < dif, order, self.pop_size)
            selected = selected.at[idx].set(False)
            return selected, knee, fitness, mask, plane, dif

        def too_few(info):
            selected, knee, fitness, mask, plane, dif = info
            mask = ~knee & mask
            fitness = jnp.where(
                jnp.repeat(mask[:, None], self.n_objs, axis=1), fitness, jnp.nan
            )
            order = jnp.argsort(plane @ fitness.T)
            idx = jnp.where(jnp.arange(len(order)) < dif, order, self.pop_size)
            selected = selected.at[idx].set(True)
            return selected, knee, fitness, mask, plane, dif

        selected = (rank < last_rank) | knee
        dif = jnp.sum(selected) - self.pop_size
        info = (selected, knee, ranked_fitness, rank == last_rank, last_plane, abs(dif))
        selected, _, _, _, _, _ = jax.lax.cond(dif > 0, too_many, lambda x: x, info)
        selected, _, _, _, _, _ = jax.lax.cond(dif < 0, too_few, lambda x: x, info)
        idx = jnp.where(selected, jnp.arange(len(selected)), len(selected)).sort()[
            : self.pop_size
        ]
        state = state.update(
            population=ranked_pop[idx],
            fitness=ranked_fitness[idx],
            knee=knee[idx],
            r=r,
            t=t,
        )
        return state
