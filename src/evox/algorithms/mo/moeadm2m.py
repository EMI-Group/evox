# --------------------------------------------------------------------------------------
# 1. MOEA/D based on MOP to MOP algorithm is described in the following papers:
#
# Title: Decomposition of a Multiobjective Optimization Problem Into a Number of Simple Multiobjective Subproblems
# Link: https://ieeexplore.ieee.org/abstract/document/6595549
#
# 2. This code has been inspired by PlatEMO.
# More information about PlatEMO can be found at the following URL:
# GitHub Link: https://github.com/BIMK/PlatEMO
# --------------------------------------------------------------------------------------

import jax
import jax.numpy as jnp

from evox import jit_class, Algorithm, State
from evox.operators import sampling, non_dominated_sort, crowding_distance
from evox.utils import cos_dist
from functools import partial


@jit_class
class Crossover:
    def __call__(self, key, p1, p2, scale):
        n, d = jnp.shape(p1)

        subkey1, subkey2 = jax.random.split(key)
        rc = (2 * jax.random.uniform(subkey1, (n, 1)) - 1) * (
            1 - jax.random.uniform(subkey2, (n, 1))
        ) ** (-((1 - scale) ** 0.7))
        offspring = p1 + jnp.tile(rc, (1, d)) * (p1 - p2)
        return offspring


@jit_class
class Mutation:
    def __call__(self, key, p1, off, scale, lb, ub):
        n, d = jnp.shape(p1)
        subkey1, subkey2, subkey3, subkey4 = jax.random.split(key, 4)
        rm = (
            0.25
            * (2 * jax.random.uniform(subkey1, (n, d)) - 1)
            * (1 - jax.random.uniform(subkey2, (n, d))) ** (-((1 - scale) ** 0.7))
        )
        site = jax.random.uniform(subkey3, (n, d)) < (1 / d)
        lower = jnp.tile(lb, (n, 1))
        upper = jnp.tile(ub, (n, 1))
        offspring = jnp.where(site, off + rm * (upper - lower), off)
        rnd = jax.random.uniform(subkey4, (n, d))
        offspring = jnp.where(
            offspring < lower, lower + 0.5 * rnd * (p1 - lower), offspring
        )
        offspring = jnp.where(
            offspring > upper, upper - 0.5 * rnd * (upper - p1), offspring
        )
        return offspring


@partial(jax.jit, static_argnums=4)
def associate(rng, pop, obj, w, s):
    k = len(w)
    dis = cos_dist(obj, w)
    max_indices = jnp.argmax(dis, axis=1)
    partition = jnp.zeros((s, k), dtype=int)

    def body_fun(i, p):
        mask = max_indices == i
        current = jnp.where(mask, size=len(pop), fill_value=-1)[0]

        def true_fun(c):
            c = c[:s]
            rad = jax.random.randint(rng, (s,), 0, len(pop))
            c = jnp.where(c != -1, c, rad)
            return c

        def false_fun(c):
            rank = non_dominated_sort(obj)
            rank = jnp.where(mask, rank, jnp.inf)
            order = jnp.argsort(rank)
            worst_rank = rank[order[s - 1]]
            mask_worst = rank == worst_rank
            crowding_dis = crowding_distance(obj, mask_worst)
            c = jnp.lexsort((-crowding_dis, rank))[:s]
            return c

        current = jax.lax.cond(jnp.sum(mask) < s, true_fun, false_fun, current)
        p = p.at[:, i].set(current)
        return p

    partition = jax.lax.fori_loop(0, k, body_fun, partition)

    partition = partition.flatten(order="F")
    return pop[partition], obj[partition]


@jit_class
class MOEADM2M(Algorithm):
    """MOEA/D based on MOP to MOP algorithm

    link: https://ieeexplore.ieee.org/abstract/document/6595549
    """

    def __init__(
        self,
        lb,
        ub,
        n_objs,
        pop_size,
        k=10,
        max_gen=100,
        mutation_op=None,
        crossover_op=None,
    ):
        self.lb = lb
        self.ub = ub
        self.n_objs = n_objs
        self.dim = lb.shape[0]
        self.k = k
        self.pop_size = (jnp.ceil(pop_size / self.k) * self.k).astype(int)
        self.s = int(self.pop_size / self.k)
        self.max_gen = max_gen

        self.mutation = Mutation()
        self.crossover = Crossover()

        self.sample = sampling.LatinHypercubeSampling(self.k, self.n_objs)

    def setup(self, key):
        key, subkey1, subkey2 = jax.random.split(key, 3)
        population = (
            jax.random.uniform(subkey1, shape=(self.pop_size, self.dim))
            * (self.ub - self.lb)
            + self.lb
        )

        w = self.sample(subkey2)[0]

        return State(
            population=population,
            fitness=jnp.zeros((self.pop_size, self.n_objs)),
            next_generation=population,
            w=w,
            is_init=True,
            key=key,
            gen=0,
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
        key, local_key, global_key, rnd_key, x_key, mut_key = jax.random.split(
            state.key, 6
        )
        current_gen = state.gen
        scale = current_gen / self.max_gen
        population = state.population
        mating_pool_local = jax.random.randint(
            local_key, (self.s, self.k), 0, self.s
        ) + jnp.tile(jnp.arange(0, self.s * self.k, self.s), (self.s, 1))
        mating_pool_local = mating_pool_local.flatten()
        mating_pool_global = jax.random.randint(
            global_key, (self.pop_size,), 0, self.pop_size
        )

        rnd = jax.random.uniform(rnd_key, (self.s, self.k)).flatten()

        mating_pool_local = jnp.where(rnd < 0.7, mating_pool_global, mating_pool_local)

        crossovered = self.crossover(
            x_key, population, population[mating_pool_local], scale
        )
        next_generation = self.mutation(
            mut_key, population, crossovered, scale, self.lb, self.ub
        )
        current_gen = current_gen + 1

        return next_generation, state.update(
            next_generation=next_generation, key=key, gen=current_gen
        )

    def _tell_init(self, state, fitness):
        key, subkey = jax.random.split(state.key)
        population = state.population
        population, fitness = associate(subkey, population, fitness, state.w, self.s)

        state = state.update(
            population=population, fitness=fitness, is_init=False, key=key
        )
        return state

    def _tell_normal(self, state, fitness):
        key, subkey = jax.random.split(state.key)
        merged_pop = jnp.concatenate([state.population, state.next_generation], axis=0)
        merged_fitness = jnp.concatenate([state.fitness, fitness], axis=0)

        population, pop_obj = associate(
            subkey, merged_pop, merged_fitness, state.w, self.s
        )

        state = state.update(population=population, fitness=pop_obj, key=key)
        return state
