import evox as ex
import jax
import jax.numpy as jnp

from evox.operators.selection import UniformRandomSelection
from evox.operators.mutation import GaussianMutation, PmMutation
from evox.operators.crossover import UniformCrossover, SimulatedBinaryCrossover
from evox.operators import non_dominated_sort, crowding_distance_sort


@ex.jit_class
class NSGA2(ex.Algorithm):
    """NSGA-II algorithm

    link: https://ieeexplore.ieee.org/document/996017
    """

    def __init__(
        self,
        lb,
        ub,
        n_objs,
        pop_size,
        selection=UniformRandomSelection(p=0.5),
        mutation=GaussianMutation(),
        # mutation=PmMutation(),
        crossover=UniformCrossover(),
        # crossover=SimulatedBinaryCrossover(),
    ):
        self.lb = lb
        self.ub = ub
        self.n_objs = n_objs
        self.dim = lb.shape[0]
        self.pop_size = pop_size

        self.selection = selection
        self.mutation = mutation
        self.crossover = crossover

    def setup(self, key):
        key, subkey = jax.random.split(key)
        population = (
            jax.random.uniform(subkey, shape=(self.pop_size, self.dim))
            * (self.ub - self.lb)
            + self.lb
        )
        return ex.State(
            population=population,
            fitness=jnp.zeros((self.pop_size, self.n_objs)),
            next_generation=population,
            is_init=True)

    def ask(self, state):
        return jax.lax.cond(state.is_init, self._ask_init, self._ask_normal, state)

    def tell(self, state, fitness):
        return jax.lax.cond(state.is_init,
                            self._tell_init,
                            self._tell_normal,
                            state, fitness
                            )

    def _ask_init(self, state):
        return state, state.population

    def _ask_normal(self, state):
        state, mutated = self.selection(state, state.population)
        state, mutated = self.mutation(state, mutated)
        
        state, crossovered = self.selection(state, state.population)
        state, crossovered = self.crossover(state, crossovered)
        
        # state, crossovered = self.crossover(state, state.population)
        # state, next_generation = self.mutation(state, crossovered, (self.lb, self.ub))
        next_generation = jnp.clip(
            jnp.concatenate([mutated, crossovered], axis=0), self.lb, self.ub
        )
        return state.update(next_generation=next_generation), next_generation

    def _tell_init(self, state, fitness):
        state = state.update(fitness=fitness, is_init=False)
        return state

    def _tell_normal(self, state, fitness):
        merged_pop = jnp.concatenate([state.population, state.next_generation], axis=0)
        merged_fitness = jnp.concatenate([state.fitness, fitness], axis=0)

        rank = non_dominated_sort(merged_fitness)
        order = jnp.argsort(rank)
        worst_rank = rank[order[self.pop_size]]
        mask = (rank == worst_rank)
        crowding_distance = crowding_distance_sort(merged_fitness, mask)

        combined_order = jnp.lexsort(
            (-crowding_distance, rank))[:self.pop_size]
        survivor = merged_pop[combined_order]
        survivor_fitness = merged_fitness[combined_order]
        state = state.update(population=survivor, fitness=survivor_fitness)
        return state
