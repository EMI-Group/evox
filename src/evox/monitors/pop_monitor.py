from typing import Any, Optional
import warnings
import time

import jax
import jax.numpy as jnp

from evox import Monitor, pytree_field, dataclass
from evox.vis_tools import plot
from .evoxvision_adapter import EvoXVisionAdapter, new_exv_metadata


@dataclass
class PopMonitorState:
    first_step: bool = pytree_field(static=True)
    latest_population: jax.Array
    latest_fitness: jax.Array


@dataclass
class PopMonitor(Monitor):
    """Population monitor,
    used to monitor the population inside the genetic algorithm.

    Parameters
    ----------
    population_name
        The name of the population in the state.
        Default to "population".
    fitness_name
        The name of the fitness in the state.
        Default to "fitness".
    to_host
        Whether to move the population and fitness to host memory (ram).
        Doing so can reduce memory usage on device (vram),
        but also introduces overhead of data transfer.
        Default to False.
    fitness_only
        Whether to only record the fitness.
        Setting it to True will disable the recording of population (decision space),
        only the fitness (objective space) will be recorded.
        This can reduce memory usage if you only care about the fitness.
        Default to False.
    """

    population_name: str = pytree_field(default="population", static=True)
    fitness_name: str = pytree_field(default="fitness", static=True)
    full_fit_history: bool = pytree_field(default=True, static=True)
    full_pop_history: bool = pytree_field(default=True, static=True)

    opt_direction: int = pytree_field(static=True, init=False, default=1)
    fitness_history: list = pytree_field(static=True, init=False, default_factory=list)
    population_history: list = pytree_field(
        static=True, init=False, default_factory=list
    )
    timestamp_history: list = pytree_field(
        static=True, init=False, default_factory=list
    )
    evoxvision_adapter: Optional[EvoXVisionAdapter] = pytree_field(
        default=None, static=True
    )

    def hooks(self):
        return ["post_step"]

    def set_opt_direction(self, opt_direction):
        self.set_frozen_attr("opt_direction", opt_direction)

    def setup(self, key: jax.Array) -> Any:
        return PopMonitorState(
            first_step=True,
            latest_population=None,
            latest_fitness=None,
        )

    def clear_history(self):
        """Clear the history of fitness and solutions.
        Normally it will be called at the initialization of Workflow object."""
        self.fitness_history = []
        self.population_history = []

    def post_step(self, state, workflow_state):
        algorithm_state = workflow_state.get_child_state("algorithm")
        population = getattr(algorithm_state, self.population_name)
        fitness = getattr(algorithm_state, self.fitness_name)

        state = state.replace(
            latest_population=population,
            latest_fitness=fitness,
        )
        if state.first_step:
            state = state.replace(first_step=False)

        return state.register_callback(self._record_history, population, fitness)

    def _record_history(self, population, fitness):
        if self.evoxvision_adapter:
            if not self.evoxvision_adapter.header_written:
                # wait for the first two iterations
                self.population_history.append(jax.device_get(population))
                self.fitness_history.append(jax.device_get(fitness))
                if len(self.population_history) >= 2:
                    metadata = new_exv_metadata(
                        self.population_history[0],
                        self.population_history[1],
                        self.fitness_history[0],
                        self.fitness_history[1],
                    )
                    self.evoxvision_adapter.set_metadata(metadata)
                    self.evoxvision_adapter.write_header()
                    self.evoxvision_adapter.write(
                        self.population_history[0].tobytes(),
                        self.fitness_history[0].tobytes(),
                    )
                    self.evoxvision_adapter.write(
                        self.population_history[1].tobytes(),
                        self.fitness_history[1].tobytes(),
                    )
                    self.population_history = []
                    self.fitness_history = []
            else:
                self.evoxvision_adapter.write(population.tobytes(), fitness.tobytes())
        else:
            if self.full_pop_history:
                self.population_history.append(population)
            if self.full_fit_history:
                self.fitness_history.append(fitness)
            self.timestamp_history.append(time.time())

    def plot(self, state=None, problem_pf=None, **kwargs):
        if not self.fitness_history:
            warnings.warn("No fitness history recorded, return None")
            return

        if self.fitness_history[0].ndim == 1:
            n_objs = 1
        else:
            n_objs = self.fitness_history[0].shape[1]

        if n_objs == 1:
            return plot.plot_obj_space_1d(self.fitness_history, **kwargs)
        elif n_objs == 2:
            return plot.plot_obj_space_2d(self.fitness_history, problem_pf, **kwargs)
        elif n_objs == 3:
            return plot.plot_obj_space_3d(self.fitness_history, problem_pf, **kwargs)
        else:
            warnings.warn("Not supported yet.")

    def get_latest_fitness(self, state):
        return state.latest_fitness, state

    def get_latest_population(self, state):
        return state.latest_population, state

    def get_population_history(self, state=None):
        return self.population_history, state

    def get_fitness_history(self, state=None):
        return self.fitness_history, state
