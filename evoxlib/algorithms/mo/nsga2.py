import evoxlib as exl
import jax
import jax.numpy as jnp

from evoxlib.operators.selection import UniformRandomSelection
from evoxlib.operators.mutation import GaussianMutation
from evoxlib.operators.crossover import UniformCrossover
from evoxlib.operators import non_dominated_sort, crowding_distance_sort


@exl.jit_class
class NSGA2(exl.Algorithm):
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
        crossover=UniformCrossover(),
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
        return exl.State(population=population, costs=jnp.zeros((self.pop_size, self.n_objs)), is_init=True)

    @exl.jit_method
    def ask(self, state):
        return jax.lax.cond(state.is_init, self._ask_init, self._ask_normal, state)

    def tell(self, state, pop, costs):
        return jax.lax.cond(state.is_init,
                            self._tell_init,
                            self._tell_normal,
                            state, pop, costs
                            )

    def _ask_init(self, state):
        return state, state.population

    def _ask_normal(self, state):
        state, mutated = self.selection(state, state.population)
        state, mutated = self.mutation(state, mutated)

        state, crossovered = self.selection(state, state.population)
        state, crossovered = self.crossover(state, crossovered)

        return state, jnp.clip(
            jnp.concatenate([mutated, crossovered], axis=0), self.lb, self.ub
        )

    def _tell_init(self, state, pop, costs):
        state = state.update(costs=costs, is_init=False)
        return state

    def _tell_normal(self, state, pop, costs):
        merged_pop = jnp.concatenate([state.population, pop], axis=0)
        merged_costs = jnp.concatenate([state.costs, costs], axis=0)

        rank = non_dominated_sort(merged_costs)
        order = jnp.argsort(rank)
        worst_rank = rank[order[self.pop_size]]
        mask = (rank == worst_rank)
        crowding_distance = crowding_distance_sort(merged_costs, mask)

        combined_order = jnp.lexsort(
            (-crowding_distance, rank))[:self.pop_size]
        survivor = merged_pop[combined_order]
        survivor_costs = merged_costs[combined_order]

        state = state.update(population=survivor, costs=survivor_costs)
        return state
