# --------------------------------------------------------------------------------------
# 1. SPEA2 algorithm is described in the following papers:
#
# Title: SPEA2: Improving the strength pareto evolutionary algorithm
# Link: https://www.research-collection.ethz.ch/handle/20.500.11850/145755
#
# 2. This code has been inspired by PlatEMO.
# More information about PlatEMO can be found at the following URL:
# GitHub Link: https://github.com/BIMK/PlatEMO
# --------------------------------------------------------------------------------------

import jax
import jax.numpy as jnp

from evox import jit_class, Algorithm, State
from evox.operators import selection, mutation, crossover
from evox.utils import dominate_relation, pairwise_euclidean_dist


@jax.jit
def cal_fitness(obj):
    n = jnp.shape(obj)[0]

    dom_matrix = dominate_relation(obj, obj)
    s = jnp.sum(dom_matrix, axis=1)

    r = jax.vmap(
        lambda s, d: jnp.sum(jnp.where(d, s, 0)), in_axes=(None, 1), out_axes=0
    )(s, dom_matrix)

    dis = pairwise_euclidean_dist(obj, obj)
    diagonal_indices = jnp.arange(n)
    dis = jnp.where(diagonal_indices[:, None] == diagonal_indices, jnp.inf, dis)
    dis = jnp.sort(dis, axis=1)

    d = 1 / (dis[:, jnp.floor(jnp.sqrt(6)).astype(int) - 1] + 2)
    return d + r


@jax.jit
def _truncation(obj, k, mask):
    n = jnp.shape(obj)[0]
    dis = pairwise_euclidean_dist(obj, obj)
    diagonal_indices = jnp.arange(n)
    dis = jnp.where(diagonal_indices[:, None] == diagonal_indices, jnp.inf, dis)
    mask_matrix = mask[:, jnp.newaxis] & mask[jnp.newaxis, :]

    dis = jnp.where(mask_matrix, dis, jnp.inf)
    del_ind = jnp.ones(n, dtype=jnp.bool_)

    def cond_fun(vals):
        del_ind, _ = vals
        return jnp.sum(~del_ind) < k

    def body_fun(vals):
        del_ind, dis = vals
        temp = jnp.min(dis, axis=1)
        rank = jnp.argsort(temp)
        idx = rank[0]
        del_ind = del_ind.at[idx].set(False)
        dis = dis.at[idx, :].set(jnp.inf)
        dis = dis.at[:, idx].set(jnp.inf)
        return (del_ind, dis)

    del_ind, dis = jax.lax.while_loop(cond_fun, body_fun, (del_ind, dis))

    return del_ind & mask


@jit_class
class SPEA2(Algorithm):
    """SPEA2 algorithm

    link: https://www.research-collection.ethz.ch/handle/20.500.11850/145755
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
        self.selection = selection.Tournament(n_round=self.pop_size)

        if self.mutation is None:
            self.mutation = mutation.Polynomial((lb, ub))
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
            is_init=True,
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
        population = state.population

        sig_fitness = cal_fitness(state.fitness)

        selected, _ = self.selection(sel_key, population, sig_fitness)
        crossovered = self.crossover(x_key, selected)
        next_generation = self.mutation(mut_key, crossovered)

        return next_generation, state.update(next_generation=next_generation)

    def _tell_init(self, state, fitness):
        state = state.update(fitness=fitness, is_init=False)
        return state

    def _tell_normal(self, state, fitness):
        merged_pop = jnp.concatenate([state.population, state.next_generation], axis=0)
        merged_fitness = jnp.concatenate([state.fitness, fitness], axis=0)

        sig_fitness = cal_fitness(merged_fitness)
        mask = sig_fitness < 1
        num_valid = jnp.sum(mask)

        def fitness_sort(mask):
            order = jnp.argsort(sig_fitness)
            return order

        def truncation(mask):
            order = _truncation(merged_fitness, num_valid - self.pop_size, mask)
            order = jnp.where(order, size=len(mask))[0]
            return order

        order = jax.lax.cond(num_valid <= self.pop_size, fitness_sort, truncation, mask)

        combined_order = order[: self.pop_size]

        survivor = merged_pop[combined_order]
        survivor_fitness = merged_fitness[combined_order]
        state = state.update(population=survivor, fitness=survivor_fitness)
        return state
