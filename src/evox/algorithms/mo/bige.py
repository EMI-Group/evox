# --------------------------------------------------------------------------------------
# 1. BiGE algorithm is described in the following papers:
#
# Title: Bi-goal evolution for many-objective optimization problems
# Link: https://doi.org/10.1016/j.artint.2015.06.007
#
# 2. This code has been inspired by PlatEMO.
# More information about PlatEMO can be found at the following URL:
# GitHub Link: https://github.com/BIMK/PlatEMO
# --------------------------------------------------------------------------------------

import jax
import jax.numpy as jnp
from jax import vmap

from evox.operators import (
    selection,
    mutation,
    crossover,
    non_dominated_sort,
)
from evox import Algorithm, jit_class, State
from evox.utils import euclidean_dist


@jax.jit
def estimate(fit, mask):
    # calculate proximity and crowding degree as bi-goals
    def calc_sh(fit_a, pr_a, fit_b, pr_b, r):
        dis = euclidean_dist(fit_a, fit_b)
        res = (
            (dis < r) * 0.5 * ((1 + (pr_a >= pr_b) + (pr_a > pr_b)) * (1 - dis / r))
        ) ** 2
        return res

    n, m = jnp.sum(mask), fit.shape[1]
    r = 1 / n ** (1 / m)
    fit_mask = mask[:, None].repeat(m, axis=1)
    fit = jnp.where(fit_mask, fit, jnp.nan)
    f_max = jnp.nanmax(fit, axis=0)
    f_min = jnp.nanmin(fit, axis=0)
    normed_fit = (fit - f_min) / (f_max - f_min).clip(1e-6)
    normed_fit = jnp.where(fit_mask, normed_fit, m)

    # pr: proximity
    # sh: sharing function
    # cd: crowding degree
    pr = jnp.sum(normed_fit, axis=1)
    sh = vmap(
        lambda _f_a, _pr_a, _r: vmap(
            lambda _f_b, _pr_b: calc_sh(_f_a, _pr_a, _f_b, _pr_b, _r)
        )(normed_fit, pr),
        (0, 0, None),
    )(normed_fit, pr, r)
    cd = jnp.sqrt(jnp.sum(sh, axis=1) - sh.diagonal())

    bi_fit = jnp.hstack([pr[:, None], cd[:, None]])
    bi_mask = mask[:, None].repeat(2, axis=1)
    bi_fit = jnp.where(bi_mask, bi_fit, jnp.inf)
    return bi_fit


@jit_class
class BiGE(Algorithm):
    """BiGE algorithm
    
    link: https://doi.org/10.1016/j.artint.2015.06.007
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
        bi_fit = estimate(state.population, jnp.full((self.pop_size,), True))
        bi_rank = non_dominated_sort(bi_fit)

        keys = jax.random.split(state.key, 4)
        selected, _ = self.selection(keys[1], state.population, bi_rank)
        crossovered = self.crossover(keys[2], selected)
        mutated = self.mutation(keys[3], crossovered)
        next_gen = jnp.clip(mutated, self.lb, self.ub)
        return next_gen, state.update(next_generation=next_gen, key=keys[0])

    def _tell_init(self, state, fitness):
        state = state.update(fitness=fitness, is_init=False)
        return state

    def _tell_normal(self, state, fitness):
        merged_pop = jnp.concatenate([state.population, state.next_generation], axis=0)
        merged_fit = jnp.concatenate([state.fitness, fitness], axis=0)
        rank = non_dominated_sort(merged_fit)
        order = jnp.argsort(rank)
        rank = rank[order]
        ranked_pop = merged_pop[order]
        ranked_fit = merged_fit[order]
        last_rank = rank[self.pop_size]

        bi_fit = estimate(ranked_fit, rank == last_rank)
        bi_rank = non_dominated_sort(bi_fit)

        fin_rank = jnp.where(rank >= last_rank, bi_rank, -1)
        idx = jnp.argsort(fin_rank)[: self.pop_size]
        state = state.update(population=ranked_pop[idx], fitness=ranked_fit[idx])
        return state
