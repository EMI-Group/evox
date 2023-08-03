import jax
import jax.numpy as jnp
from functools import partial

<<<<<<< HEAD
from evox import jit_class, Algorithm, State
from evox.operators.selection import TournamentSelection, non_dominated_sort, crowding_distance
from evox.operators.mutation import PmMutation
from evox.operators.crossover import SimulatedBinaryCrossover
from jax.experimental.host_callback import id_print


@partial(jax.jit, static_argnames=['n'])
def environmental_selection(fitness, n):
    rank = non_dominated_sort(fitness)
    order = jnp.argsort(rank)
    worst_rank = rank[order[n - 1]]
    mask = rank == worst_rank
    crowding_dis = crowding_distance(fitness, mask)
    combined_indices = jnp.lexsort((-crowding_dis, rank))[: n]

    return combined_indices


=======
from evox.operators import (
    non_dominated_sort,
    crowding_distance_sort,
    selection,
    mutation,
    crossover,
)
from evox import Algorithm, jit_class, State


>>>>>>> e9d1a7a9a7ff3bb82fc0c14c4cd4180929c822b9
@jit_class
class NSGA2(Algorithm):
    """NSGA-II algorithm

    link: https://ieeexplore.ieee.org/document/996017
    """

    def __init__(
<<<<<<< HEAD
            self,
            lb,
            ub,
            n_objs,
            pop_size,

            mutation=PmMutation(),
            crossover=SimulatedBinaryCrossover(),
=======
        self,
        lb,
        ub,
        n_objs,
        pop_size,
        selection_op=None,
        mutation_op=None,
        crossover_op=None,
>>>>>>> e9d1a7a9a7ff3bb82fc0c14c4cd4180929c822b9
    ):
        self.lb = lb
        self.ub = ub
        self.n_objs = n_objs
        self.dim = lb.shape[0]
        self.pop_size = pop_size

<<<<<<< HEAD
        self.selection = TournamentSelection(num_round=self.pop_size)
        self.mutation = mutation
        self.crossover = crossover
=======
        self.selection = selection_op
        self.mutation = mutation_op
        self.crossover = crossover_op

        if self.selection is None:
            self.selection = selection.UniformRand(0.5)
        if self.mutation is None:
            self.mutation = mutation.Gaussian()
        if self.crossover is None:
            self.crossover = crossover.UniformRand()
>>>>>>> e9d1a7a9a7ff3bb82fc0c14c4cd4180929c822b9

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
<<<<<<< HEAD
        crossovered, state = self.crossover(state, state.population)
        next_generation, state = self.mutation(state, crossovered, (self.lb, self.ub))

        return next_generation, state.update(next_generation=next_generation)
=======
        key, sel_key1, mut_key, sel_key2, x_key = jax.random.split(state.key, 5)
        mutated = self.selection(sel_key1, state.population)
        mutated = self.mutation(mut_key, mutated)

        crossovered = self.selection(sel_key2, state.population)
        crossovered = self.crossover(x_key, crossovered)

        next_generation = jnp.clip(
            jnp.concatenate([mutated, crossovered], axis=0), self.lb, self.ub
        )
        return next_generation, state.update(next_generation=next_generation, key=key)
>>>>>>> e9d1a7a9a7ff3bb82fc0c14c4cd4180929c822b9

    def _tell_init(self, state, fitness):
        state = state.update(fitness=fitness, is_init=False)
        return state

    def _tell_normal(self, state, fitness):
        merged_pop = jnp.concatenate([state.population, state.next_generation], axis=0)
        merged_fitness = jnp.concatenate([state.fitness, fitness], axis=0)

        combined_order = environmental_selection(merged_fitness, self.pop_size)

        survivor = merged_pop[combined_order]
        survivor_fitness = merged_fitness[combined_order]
        state = state.update(population=survivor, fitness=survivor_fitness)
        return state
