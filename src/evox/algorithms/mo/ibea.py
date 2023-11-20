# --------------------------------------------------------------------------------------
# 1. IBEA algorithm is described in the following papers:
#
# Title: Indicator-Based Selection in Multiobjective Search
# Link: https://link.springer.com/chapter/10.1007/978-3-540-30217-9_84
#
# 2. This code has been inspired by PlatEMO.
# More information about PlatEMO can be found at the following URL:
# GitHub Link: https://github.com/BIMK/PlatEMO
# --------------------------------------------------------------------------------------

import jax
import jax.numpy as jnp

from evox.operators import selection, mutation, crossover
from evox.utils import cal_max
from evox import Algorithm, State, jit_class


@jax.jit
def cal_fitness(pop_obj, kappa):
    n = jnp.shape(pop_obj)[0]
    pop_obj = (pop_obj - jnp.tile(jnp.min(pop_obj), (n, 1))) / (
        jnp.tile(jnp.max(pop_obj) - jnp.min(pop_obj), (n, 1))
    )
    I = cal_max(pop_obj, pop_obj)

    C = jnp.max(jnp.abs(I), axis=0)

    fitness = jnp.sum(-jnp.exp(-I / jnp.tile(C, (n, 1)) / kappa), axis=0) + 1

    return fitness, I, C


@jit_class
class IBEA(Algorithm):
    """IBEA algorithm

    link: https://link.springer.com/chapter/10.1007/978-3-540-30217-9_84

    Args:
        kappa: fitness scaling factor. Default: 0.05
    """

    def __init__(
        self,
        lb,
        ub,
        n_objs,
        pop_size,
        kappa=0.05,
        mutation_op=None,
        crossover_op=None,
    ):
        self.lb = lb
        self.ub = ub
        self.n_objs = n_objs
        self.dim = lb.shape[0]
        self.pop_size = pop_size
        self.kappa = kappa

        self.selection = selection.Tournament(n_round=self.pop_size)
        self.mutation = mutation_op
        self.crossover = crossover_op

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
        )

    def init_ask(self, state):
        return state.population, state

    def init_tell(self, state, fitness):
        state = state.update(fitness=fitness)
        return state

    def ask(self, state):
        key, sel_key, x_key, mut_key = jax.random.split(state.key, 4)
        population = state.population
        pop_obj = state.fitness
        fitness = cal_fitness(pop_obj, self.kappa)[0]

        selected, _ = self.selection(sel_key, population, -fitness)
        crossovered = self.crossover(x_key, selected)
        next_generation = self.mutation(mut_key, crossovered)

        return next_generation, state.update(next_generation=next_generation, key=key)

    def tell(self, state, fitness):
        merged_pop = jnp.concatenate([state.population, state.next_generation], axis=0)
        merged_obj = jnp.concatenate([state.fitness, fitness], axis=0)

        merged_fitness, I, C = cal_fitness(merged_obj, self.kappa)

        # Different from the original paper, the selection here is directly through fitness.
        next_ind = jnp.argsort(-merged_fitness)[0: self.pop_size]

        # The following code is from the original paper's implementation
        # and is kept for reference purposes but is not being used in this version.
        # n = jnp.shape(merged_pop)[0]
        # next_ind = jnp.arange(n)
        # vals = (next_ind, merged_fitness)
        # def body_fun(i, vals):
        #     next_ind, merged_fitness = vals
        #     x = jnp.argmin(merged_fitness)
        #     merged_fitness += jnp.exp(-I[x, :] / C[x] / self.kappa)
        #     merged_fitness = merged_fitness.at[x].set(jnp.max(merged_fitness))
        #     next_ind = next_ind.at[x].set(-1)
        #     return (next_ind, merged_fitness)
        #
        # next_ind, merged_fitness = jax.lax.fori_loop(0, self.pop_size, body_fun, vals)
        #
        # next_ind = jnp.where(next_ind != -1, size=n, fill_value=-1)[0]
        # next_ind = next_ind[0: self.pop_size]

        survivor = merged_pop[next_ind]
        survivor_fitness = merged_obj[next_ind]

        state = state.update(population=survivor, fitness=survivor_fitness)

        return state
