from typing import Any
import warnings

import jax
import jax.numpy as jnp

from evox import Monitor
from evox.vis_tools import plot


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

    def __init__(
        self,
        population_name="population",
        fitness_name="fitness",
        to_host=False,
        fitness_only=False,
    ):
        super().__init__()
        self.population_name = population_name
        self.fitness_name = fitness_name
        self.to_host = to_host
        if to_host:
            self.host = jax.devices("cpu")[0]
        self.population_history = []
        self.fitness_history = []
        self.fitness_only = fitness_only

    def hooks(self):
        return ["post_step"]

    def post_step(self, state):
        if not self.fitness_only:
            population = getattr(
                state.get_child_state("algorithm"), self.population_name
            )
            if self.to_host:
                population = jax.device_put(population, self.host)
            self.population_history.append(population)

        fitness = getattr(state.get_child_state("algorithm"), self.fitness_name)
        if self.to_host:
            fitness = jax.device_put(fitness, self.host)
        self.fitness_history.append(fitness)

    def plot(self, problem_pf=None, **kwargs):
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

    def get_population_history(self):
        return self.population_history

    def get_fitness_history(self):
        return self.fitness_history
