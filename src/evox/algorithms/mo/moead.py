# --------------------------------------------------------------------------------------
# 1. MOEA/D algorithm is described in the following papers:
#
# Title: MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition
# Link: https://ieeexplore.ieee.org/document/4358754
# --------------------------------------------------------------------------------------

import math

import jax
import jax.numpy as jnp

from evox import Algorithm, State, jit_class
from evox.operators import crossover, mutation
from evox.operators.sampling import UniformSampling
from evox.utils import pairwise_euclidean_dist, AggregationFunction

@jit_class
class MOEAD(Algorithm):
    """Parallel MOEA/D algorithm

    link: https://ieeexplore.ieee.org/document/4358754
    """

    def __init__(
        self,
        lb,
        ub,
        n_objs,
        pop_size,
        func_name='pbi',
        mutation_op=None,
        crossover_op=None,
    ):
        self.lb = lb
        self.ub = ub
        self.n_objs = n_objs
        self.dim = lb.shape[0]
        self.pop_size = pop_size
        self.func_name = func_name
        self.n_neighbor = 0

        self.mutation = mutation_op
        self.crossover = crossover_op

        if self.mutation is None:
            self.mutation = mutation.Polynomial((lb, ub))
        if self.crossover is None:
            self.crossover = crossover.SimulatedBinary(type=2)
        self.sample = UniformSampling(self.pop_size, self.n_objs)
        self.aggregate_func = AggregationFunction(self.func_name)

    def setup(self, key):
        key, subkey1, subkey2 = jax.random.split(key, 3)
        w, _ = self.sample(subkey2)
        self.pop_size = w.shape[0]
        self.n_neighbor = int(math.ceil(self.pop_size / 10))

        population = (
            jax.random.uniform(subkey1, shape=(self.pop_size, self.dim))
            * (self.ub - self.lb)
            + self.lb
        )

        neighbors = pairwise_euclidean_dist(w, w)
        neighbors = jnp.argsort(neighbors, axis=1)
        neighbors = neighbors[:, : self.n_neighbor]

        return State(
            population=population,
            fitness=jnp.zeros((self.pop_size, self.n_objs)),
            next_generation=population,
            weight_vector=w,
            neighbors=neighbors,
            z=jnp.zeros(shape=self.n_objs),
            key=key,
        )

    def init_ask(self, state):
        return state.population, state

    def init_tell(self, state, fitness):
        z = jnp.min(fitness, axis=0)
        state = state.replace(fitness=fitness, z=z)
        return state

    def ask(self, state):
        key, subkey, sel_key, mut_key = jax.random.split(state.key, 4)
        parent = jax.random.permutation(
            subkey, state.neighbors, axis=1, independent=True
        ).astype(int)

        population = state.population
        selected_p = jnp.r_[population[parent[:, 0]], population[parent[:, 1]]]

        crossovered = self.crossover(sel_key, selected_p)
        next_generation = self.mutation(mut_key, crossovered)
        next_generation = jnp.clip(next_generation, self.lb, self.ub)

        return next_generation, state.replace(
            next_generation=next_generation, key=key
        )

    def tell(self, state, fitness):
        population = state.population
        pop_obj = state.fitness
        offspring = state.next_generation
        obj = fitness
        w = state.weight_vector
        z = jnp.minimum(state.z, jnp.min(obj, axis=0))
        z_max = jnp.max(pop_obj, axis=0)
        neighbors = state.neighbors

        def scan_body(carry, x):
            population, pop_obj = carry
            off_pop, off_obj, indices = x

            f_old = self.aggregate_func(pop_obj[indices], w[indices], z, z_max)
            f_new = self.aggregate_func(off_obj[jnp.newaxis, :], w[indices], z, z_max)

            update_condition = (f_old > f_new)[:, jnp.newaxis]
            updated_population = population.at[indices].set(
                jnp.where(update_condition, jnp.tile(off_pop, (jnp.shape(indices)[0], 1)), population[indices]))
            updated_pop_obj = pop_obj.at[indices].set(
                jnp.where(update_condition, jnp.tile(off_obj, (jnp.shape(indices)[0], 1)), pop_obj[indices]))

            return (updated_population, updated_pop_obj), None

        (population, pop_obj), _ = jax.lax.scan(scan_body, (population, pop_obj), (offspring, obj, neighbors))


        state = state.replace(population=population, fitness=pop_obj, z=z)
        return state
