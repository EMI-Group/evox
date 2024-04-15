# --------------------------------------------------------------------------------------
# 1. RVEA algorithm is described in the following papers:
#
# Title: A Reference Vector Guided Evolutionary Algorithm for Many-Objective Optimization
# Link: https://ieeexplore.ieee.org/document/7386636
# --------------------------------------------------------------------------------------

import jax
import jax.numpy as jnp

from evox.operators import mutation, crossover, selection
from evox.operators.sampling import UniformSampling
from evox import Algorithm, State, jit_class
from evox.utils import cos_dist
from evox.operators import non_dominated_sort


@jax.jit
def rv_regeneration(pop_obj, v, key):
    """
    Regenerate reference vectors regenerate strategy.
    """
    pop_obj = pop_obj - jnp.nanmin(pop_obj, axis=0)
    cosine = cos_dist(pop_obj, v)

    associate = jnp.nanargmax(cosine, axis=1)

    invalid = jnp.sum(associate[:, jnp.newaxis] == jnp.arange(v.shape[0]), axis=0)
    rand = jax.random.uniform(key, (v.shape[0], v.shape[1])) * jnp.nanmax(
        pop_obj, axis=0
    )
    v = jnp.where(invalid[:, jnp.newaxis] == 0, rand, v)

    return v


@jax.jit
def batch_truncation(pop, obj):
    """
    Use the batch truncation operator to select the best n solutions.
    """
    n = jnp.shape(pop)[0] // 2
    cosine = cos_dist(obj, obj)
    not_all_nan_rows = ~jnp.isnan(cosine).all(axis=1)
    mask = jnp.eye(jnp.shape(cosine)[0], dtype=bool) & not_all_nan_rows[:, None]
    cosine = jnp.where(mask, 0, cosine)

    sorted_indices = jnp.sort(-cosine, axis=1)
    rank = jnp.argsort(
        jnp.where(jnp.isnan(sorted_indices[:, 0]), -jnp.inf, sorted_indices[:, 0])
    )

    mask = jnp.ones(jnp.shape(rank)[0], dtype=bool)
    mask = mask.at[rank[:n]].set(False)[:, jnp.newaxis]

    new_pop = jnp.where(mask, pop, jnp.nan)
    new_obj = jnp.where(mask, obj, jnp.nan)

    return new_pop, new_obj


@jit_class
class RVEAa(Algorithm):
    """RVEAa algorithms (RVEA embedded with the reference vector regeneration strategy)

    link: https://ieeexplore.ieee.org/document/7386636

    Args:
        alpha : The parameter controlling the rate of change of penalty. Defaults to 2.
        fr : The frequency of reference vector adaptation. Defaults to 0.1.
        max_gen : The maximum number of generations. Defaults to 100.
        If the number of iterations is not 100, change the value based on the actual value.
    """

    def __init__(
        self,
        lb,
        ub,
        n_objs,
        pop_size,
        alpha=2,
        fr=0.1,
        max_gen=100,
        selection_op=None,
        mutation_op=None,
        crossover_op=None,
    ):
        self.lb = lb
        self.ub = ub
        self.n_objs = n_objs
        self.dim = lb.shape[0]
        self.pop_size = pop_size
        self.alpha = alpha
        self.fr = fr
        self.max_gen = max_gen

        self.selection = selection_op
        self.mutation = mutation_op
        self.crossover = crossover_op

        if self.selection is None:
            self.selection = selection.ReferenceVectorGuided()
        if self.mutation is None:
            self.mutation = mutation.Polynomial((lb, ub))
        if self.crossover is None:
            self.crossover = crossover.SimulatedBinary()

        self.sampling = UniformSampling(self.pop_size, self.n_objs)

    def setup(self, key):
        key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)

        v = self.sampling(subkey1)[0]
        v0 = v
        self.pop_size = v.shape[0]

        population0 = (
            jax.random.uniform(subkey2, shape=(self.pop_size, self.dim))
            * (self.ub - self.lb)
            + self.lb
        )
        population = jnp.concatenate(
            [
                population0,
                jnp.full(shape=(self.pop_size, self.dim), fill_value=jnp.nan),
            ],
            axis=0,
        )
        v = jnp.concatenate(
            [v, jax.random.uniform(subkey3, shape=(self.pop_size, self.n_objs))], axis=0
        )

        return State(
            population=population,
            fitness=jnp.zeros((self.pop_size * 2, self.n_objs)),
            next_generation=population0,
            reference_vector=v,
            init_v=v0,
            key=key,
            gen=0,
        )

    def init_ask(self, state):
        return state.population, state

    def init_tell(self, state, fitness):
        state = state.update(fitness=fitness)
        return state

    def ask(self, state):
        key, subkey, x_key, mut_key = jax.random.split(state.key, 4)

        population = state.population
        no_nan_pop = ~jnp.isnan(population).all(axis=1)
        max_idx = jnp.sum(no_nan_pop).astype(int)
        pop = population[jnp.where(no_nan_pop, size=self.pop_size, fill_value=-1)]

        mating_pool = jax.random.randint(subkey, (self.pop_size,), 0, max_idx)
        crossovered = self.crossover(x_key, pop[mating_pool])
        next_generation = self.mutation(mut_key, crossovered)
        next_generation = jnp.clip(next_generation, self.lb, self.ub)

        return next_generation, state.update(next_generation=next_generation, key=key)

    def tell(self, state, fitness):
        key, subkey = jax.random.split(state.key, 2)
        current_gen = state.gen + 1

        v = state.reference_vector
        merged_pop = jnp.concatenate([state.population, state.next_generation], axis=0)

        merged_fitness = jnp.concatenate([state.fitness, fitness], axis=0)

        rank = non_dominated_sort(merged_fitness)
        merged_fitness = jnp.where(rank[:, jnp.newaxis] == 0, merged_fitness, jnp.nan)
        merged_pop = jnp.where(rank[:, jnp.newaxis] == 0, merged_pop, jnp.nan)

        survivor, survivor_fitness = self.selection(
            merged_pop, merged_fitness, v, (current_gen / self.max_gen) ** self.alpha
        )

        def rv_adaptation(pop_obj, v, v0):
            return v0 * (jnp.nanmax(pop_obj, axis=0) - jnp.nanmin(pop_obj, axis=0))

        def no_update(_pop_obj, v, v0):
            return v

        v_adapt = jax.lax.cond(
            current_gen % (1 / self.fr) == 0,
            rv_adaptation,
            no_update,
            survivor_fitness,
            v[: self.pop_size],
            state.init_v,
        )

        v_regen = rv_regeneration(survivor_fitness, v[self.pop_size :], subkey)
        v = jnp.concatenate([v_adapt, v_regen], axis=0)

        survivor, survivor_fitness = jax.lax.cond(
            current_gen + 1 == self.max_gen,
            batch_truncation,
            lambda x, y: (x, y),
            survivor,
            survivor_fitness,
        )

        state = state.update(
            population=survivor,
            fitness=survivor_fitness,
            reference_vector=v,
            gen=current_gen,
            key=key,
        )
        return state
