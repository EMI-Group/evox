# --------------------------------------------------------------------------------------
# 1. RVEA algorithm is described in the following papers:
#
# Title: A Reference Vector Guided Evolutionary Algorithm for Many-Objective Optimization
# Link: https://ieeexplore.ieee.org/document/7386636
# --------------------------------------------------------------------------------------

import jax
import jax.numpy as jnp

from evox.operators import mutation, crossover, selection
from evox.operators.sampling import LatinHypercubeSampling
from evox import Algorithm, State, jit_class


@jit_class
class RVEA(Algorithm):
    """RVEA algorithms

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

        self.sampling = LatinHypercubeSampling(self.pop_size, self.n_objs)

    def setup(self, key):
        key, subkey1, subkey2 = jax.random.split(key, 3)
        population = (
            jax.random.uniform(subkey1, shape=(self.pop_size, self.dim))
            * (self.ub - self.lb)
            + self.lb
        )
        v = self.sampling(subkey2)[0]
        v = v / jnp.linalg.norm(v, axis=0)

        return State(
            population=population,
            fitness=jnp.zeros((self.pop_size, self.n_objs)),
            next_generation=population,
            reference_vector=v,
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

        return next_generation, state.update(next_generation=next_generation, key=key)

    def tell(self, state, fitness):
        current_gen = state.gen + 1
        v = state.reference_vector
        merged_pop = jnp.concatenate([state.population, state.next_generation], axis=0)
        merged_fitness = jnp.concatenate([state.fitness, fitness], axis=0)

        survivor, survivor_fitness = self.selection(
            merged_pop, merged_fitness, v, (current_gen / self.max_gen) ** self.alpha
        )

        def rv_adaptation(pop_obj, v):
            v_temp = v * jnp.tile(
                (jnp.nanmax(pop_obj, axis=0) - jnp.nanmin(pop_obj, axis=0)), (len(v), 1)
            )

            next_v = v_temp / jnp.tile(
                jnp.sqrt(jnp.sum(v_temp**2, axis=1)).reshape(len(v), 1),
                (1, jnp.shape(v)[1]),
            )

            return next_v

        def no_update(_pop_obj, v):
            return v

        v = jax.lax.cond(
            current_gen % (1 / self.fr) == 0,
            rv_adaptation,
            no_update,
            survivor_fitness,
            v,
        )

        state = state.update(
            population=survivor,
            fitness=survivor_fitness,
            reference_vector=v,
            gen=current_gen,
        )
        return state
