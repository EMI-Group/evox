# --------------------------------------------------------------------------------------
# 1. Stochastic ranking algorithm is described in the following papers:
#
# Title: Stochastic Ranking Algorithm for Many-Objective Optimization Based on Multiple Indicators
# Link: https://ieeexplore.ieee.org/abstract/document/7445185
#
# 2. This code has been inspired by PlatEMO.
# More information about PlatEMO can be found at the following URL:
# GitHub Link: https://github.com/BIMK/PlatEMO
# --------------------------------------------------------------------------------------

import jax
import jax.numpy as jnp

from evox.operators import mutation, crossover
from evox.utils import cal_max
from evox import Algorithm, State, jit_class
from functools import partial


@jax.jit
def stochastic_ranking_selection(key, pc, I1, I2):
    n = len(I1)
    n_half = jnp.ceil(n / 2).astype(int)
    rank = jnp.arange(0, n)

    def swap_indices(rank, j):
        temp = rank[j]
        rank = rank.at[j].set(rank[j + 1])
        rank = rank.at[j + 1].set(temp)
        return rank

    i = 0
    swapdone = True
    rnd_all = jax.random.uniform(key, (n - 1,))

    def body_fun(vals):
        rank, swapdone, i = vals
        swapdone = False

        def in_body(j, vals):
            rank, swapdone = vals
            rnd = rnd_all[j]

            def true_fun(rank, swapdone):
                def in_true(rank, swapdone):
                    rank = swap_indices(rank, j)
                    swapdone = True
                    return rank, swapdone

                def in_false(rank, swapdone):
                    return rank, swapdone

                rank, swapdone = jax.lax.cond(
                    I1[rank[j]] < I1[rank[j + 1]], in_true, in_false, rank, swapdone
                )

                return rank, swapdone

            def false_fun(rank, swapdone):
                def in_true(rank, swapdone):
                    rank = swap_indices(rank, j)
                    swapdone = True
                    return rank, swapdone

                def in_false(rank, swapdone):
                    return rank, swapdone

                rank, swapdone = jax.lax.cond(
                    I2[rank[j]] < I2[rank[j + 1]], in_true, in_false, rank, swapdone
                )
                return rank, swapdone

            rank, swapdone = jax.lax.cond(rnd < pc, true_fun, false_fun, rank, swapdone)
            return (rank, swapdone)

        rank, swapdone = jax.lax.fori_loop(0, n - 1, in_body, (rank, swapdone))
        i += 1
        return rank, swapdone, i

    rank, s, a = jax.lax.while_loop(
        lambda vals: (vals[2] < n_half) & (vals[1]), body_fun, (rank, swapdone, i)
    )
    return rank


@partial(jax.jit, static_argnums=3)
def environmental_selection(key, pop, obj, k, uni_rnd):
    n = len(pop)

    I = cal_max(obj, obj)
    I1 = jnp.sum(-jnp.exp(-I / 0.05), axis=0) + 1

    dis = jnp.full((n, n), fill_value=jnp.inf)

    def out_body(i, val):
        s_obj = jnp.maximum(obj, jnp.tile(obj[i, :], (n, 1)))

        def in_body(j, d):
            d = d.at[i, j].set(jnp.linalg.norm(obj[i, :] - s_obj[j, :]))
            return d

        d = jax.lax.fori_loop(0, i, in_body, val)
        return d

    dis = jax.lax.fori_loop(0, n, out_body, dis)
    I2 = jnp.min(dis, axis=1)

    rank = stochastic_ranking_selection(key, uni_rnd, I1, I2)
    indices = rank[:k]
    return pop[indices], obj[indices]


@jit_class
class SRA(Algorithm):
    """Stochastic ranking algorithm

    link: https://ieeexplore.ieee.org/abstract/document/7445185
    """

    def __init__(
        self,
        lb,
        ub,
        n_objs,
        pop_size,
        mutation_op=None,
        crossover_op=None,
    ):
        self.lb = lb
        self.ub = ub
        self.n_objs = n_objs
        self.dim = lb.shape[0]
        self.pop_size = pop_size

        self.mutation = mutation_op
        self.crossover = crossover_op

        if self.mutation is None:
            self.mutation = mutation.Polynomial((lb, ub))
        if self.crossover is None:
            self.crossover = crossover.SimulatedBinary(type=2)

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
        key, sel_key, x_key, mut_key = jax.random.split(state.key, 4)

        mating_pool = jax.random.randint(
            sel_key, (self.pop_size * 2,), 0, self.pop_size
        )
        population = state.population[mating_pool]

        crossovered = self.crossover(sel_key, population)
        next_generation = self.mutation(mut_key, crossovered)

        return next_generation, state.update(next_generation=next_generation, key=key)

    def _tell_init(self, state, fitness):
        state = state.update(fitness=fitness, is_init=False)
        return state

    def _tell_normal(self, state, fitness):
        merged_pop = jnp.concatenate([state.population, state.next_generation], axis=0)
        merged_fitness = jnp.concatenate([state.fitness, fitness], axis=0)

        key, subkey, env_key = jax.random.split(state.key, 3)
        pc = jax.random.uniform(subkey) * (0.6 - 0.4) + 0.4

        population, pop_obj = environmental_selection(
            env_key, merged_pop, merged_fitness, self.pop_size, pc
        )

        state = state.update(population=population, fitness=pop_obj)
        return state
