import evoxlib as exl
import jax
import jax.numpy as jnp

from evoxlib.operators.selection import RandomSelection
from evoxlib.operators.mutation import GaussianMutation
from evoxlib.operators.crossover import UniformCrossover
from evoxlib.operators import non_dominated_sort, crowding_distance_sort

class NSGA2(exl.Algorithm):
    """NSGA-II algorithm

    link: https://ieeexplore.ieee.org/document/996017
    """

    def __init__(
        self,
        lb,
        ub,
        pop_size,
        selection=RandomSelection(p=0.5),
        mutation=GaussianMutation(),
        crossover=UniformCrossover(),
    ):
        self.lb = lb
        self.ub = ub
        self.dim = lb.shape[0]
        self.pop_size = pop_size

        self.selection = selection
        self.mutation = mutation
        self.crossover = crossover

    def setup(self, key):
        key, subkey = jax.random.split(key)
        population = jax.random.uniform(subkey, shape=(self.pop_size, self.dim)) * (self.ub - self.lb) + self.lb
        return {"population": population, "fitness": None, "is_init": True}

    @exl.jit_method
    def ask(self, state):
        return jax.lax.cond(state["is_init"], self._ask_init, self._ask_normal, state)

    def tell(self, state, x, F):
        with jax.disable_jit():
            if state["is_init"]:
                return self._tell_init(state, x, F)
            else:
                return self._tell_normal(state, x, F)

    def _ask_init(self, state):
        return state, state["population"]

    def _ask_normal(self, state):
        state, mutated = self.selection(state, state["population"])
        state, mutated = self.mutation(state, mutated)

        state, crossovered = self.selection(state, state["population"])
        state, crossovered = self.crossover(state, crossovered)

        return state, jnp.clip(jnp.concatenate([mutated, crossovered], axis=0), self.lb, self.ub)

    def _tell_init(self, state, x, F):
        state = state | {"fitness": F, "is_init": False}
        return state

    def _tell_normal(self, state, x, F):
        merged_pop = jnp.concatenate([state["population"], x], axis=0)
        merged_fitness = jnp.concatenate([state["fitness"], F], axis=0)

        rank = exl.operators.non_dominated_sort(merged_fitness)

        survivor = jnp.zeros_like(state["population"])
        survivor_fitness = jnp.zeros_like(state["fitness"])
        index = 0
        current_rank = 0
        while index < survivor.shape[0]:
            front = merged_pop[rank == current_rank]
            front_fitness = merged_fitness[rank == current_rank]
            if front.shape[0] + index <= survivor.shape[0]:
                survivor = survivor.at[index : index + front.shape[0], :].set(front)
                survivor_fitness = survivor_fitness.at[
                    index : index + front.shape[0]
                ].set(front_fitness)
                index += front.shape[0]
            else:
                crowding_distance_rank = crowding_distance_sort(front_fitness)
                survivor = survivor.at[index:, :].set(
                    front[crowding_distance_rank[index - survivor.shape[0] :], :]
                )
                survivor_fitness = survivor_fitness.at[index:].set(
                    front_fitness[crowding_distance_rank[index - survivor.shape[0] :]]
                )
                index = survivor.shape[0]
            current_rank += 1
        state = state | {"population": survivor, "fitness": survivor_fitness}
        return state
