import jax
import jax.numpy as jnp
from functools import partial

from evox import jit_class, Algorithm, State
from evox.operators import selection, mutation, crossover, non_dominated_sort


@partial(jax.jit, static_argnums=[0, 1])
def calculate_alpha(N, k):
    alpha = jnp.zeros(N)

    for i in range(1, k + 1):
        num = jnp.prod((k - jnp.arange(1, i)) / (N - jnp.arange(1, i)))
        alpha = alpha.at[i - 1].set(num / i)
    return alpha


@partial(jax.jit, static_argnums=[2, 3])
def cal_hv(points, ref, k, n_sample, key):
    n, m = jnp.shape(points)
    alpha = calculate_alpha(n, k)

    f_min = jnp.min(points, axis=0)

    s = jax.random.uniform(key, shape=(n_sample, m), minval=f_min, maxval=ref)

    pds = jnp.zeros((n, n_sample), dtype=bool)
    ds = jnp.zeros((n_sample,))

    def body_fun1(i, vals):
        pds, ds = vals
        x = jnp.sum((jnp.tile(points[i, :], (n_sample, 1)) - s) <= 0, axis=1) == m
        pds = pds.at[i].set(jnp.where(x, True, pds[i]))
        ds = jnp.where(x, ds + 1, ds)
        return pds, ds

    pds, ds = jax.lax.fori_loop(0, n, body_fun1, (pds, ds))
    ds = ds - 1

    f = jnp.zeros((n,))

    def body_fun2(pd):
        temp = jnp.where(pd, ds, -1).astype(int)
        value = jnp.where(temp != -1, alpha[temp], 0)
        value = jnp.sum(value)
        return value
    
    f = jax.vmap(body_fun2)(pds)
    f = f * jnp.prod(ref - f_min) / n_sample

    return f


@jit_class
class HypE(Algorithm):
    """HypE algorithm

    link: https://direct.mit.edu/evco/article-abstract/19/1/45/1363/HypE-An-Algorithm-for-Fast-Hypervolume-Based-Many
    Inspired by PlatEMO.
    """

    def __init__(
        self,
        lb,
        ub,
        n_objs,
        pop_size,
        n_sample=10000,
        mutation_op=None,
        crossover_op=None,
    ):
        self.lb = lb
        self.ub = ub
        self.n_objs = n_objs
        self.dim = lb.shape[0]
        self.pop_size = pop_size
        self.n_sample = n_sample

        self.mutation = mutation_op
        self.crossover = crossover_op
        self.selection = selection.Tournament(
            n_round=self.pop_size, multi_objective=True
        )
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
            ref_point=jnp.zeros((self.n_objs,)),
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
        population = state.population
        pop_obj = state.fitness
        key, subkey, sel_key, x_key, mut_key = jax.random.split(state.key, 5)
        hv = cal_hv(pop_obj, state.ref_point, self.pop_size, self.n_sample, subkey)

        selected, _ = self.selection(sel_key, population, -hv)
        crossovered = self.crossover(x_key, selected)
        next_generation = self.mutation(mut_key, crossovered)

        return next_generation, state.update(next_generation=next_generation)

    def _tell_init(self, state, fitness):
        ref_point = jnp.zeros((self.n_objs,)) + jnp.max(fitness) * 1.2
        state = state.update(fitness=fitness, ref_point=ref_point, is_init=False)
        return state

    def _tell_normal(self, state, fitness):
        merged_pop = jnp.concatenate([state.population, state.next_generation], axis=0)
        merged_obj = jnp.concatenate([state.fitness, fitness], axis=0)

        n = jnp.shape(merged_pop)[0]

        rank = non_dominated_sort(merged_obj)
        order = jnp.argsort(rank)
        worst_rank = rank[order[n - 1]]
        mask = rank == worst_rank

        key, subkey = jax.random.split(state.key)
        hv = cal_hv(merged_obj, state.ref_point, n, self.n_sample, subkey)

        dis = jnp.where(mask, hv, -jnp.inf)

        combined_indices = jnp.lexsort((-dis, rank))[: self.pop_size]

        survivor = merged_pop[combined_indices]
        survivor_fitness = merged_obj[combined_indices]

        state = state.update(population=survivor, fitness=survivor_fitness, key=key)

        return state
