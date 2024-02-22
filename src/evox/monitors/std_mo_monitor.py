import warnings

import jax
import jax.experimental.host_callback as hcb
import jax.numpy as jnp
import numpy as np
from jax.experimental import io_callback
from jax.sharding import SingleDeviceSharding

from ..operators import non_dominated_sort


class StdMOMonitor:
    """Standard multi-objective monitor
    Used for multi-objective workflow,
    can monitor fitness and record the pareto front.

    Parameters
    ----------
    record_pf
        Whether to record the pareto front during the run.
        Default to False.
        Setting it to True will cause the monitor to
        maintain a pareto front of all the solutions with unlimited size,
        which may hurt performance.
    record_fit_history
        Whether to record the full history of fitness value.
        Default to True. Setting it to False may reduce memory usage.
    """

    def __init__(
        self, record_pf=False, record_fit_history=True, record_pop_history=False
    ):
        warnings.warn(
            "The StdMOMonitor is deprecated in favor of the new EvalMonitor.",
            DeprecationWarning,
        )
        self.record_pf = record_pf
        self.record_fit_history = record_fit_history
        self.record_pop_history = record_pop_history
        self.fitness_history = []
        self.population_history = []
        self.current_population = None
        self.pf_solutions = None
        self.pf_fitness = None
        self.opt_direction = 1  # default to min, so no transformation is needed

    def set_opt_direction(self, opt_direction):
        self.opt_direction = opt_direction

    def hooks(self):
        return ["post_ask", "post_eval"]

    def post_ask(self, _state, cand_sol):
        monitor_device = SingleDeviceSharding(jax.devices()[0])
        io_callback(self.record_pop, None, cand_sol, sharding=monitor_device)

    def post_eval(self, _state, _cand_sol, _transformed_cand_sol, fitness):
        monitor_device = SingleDeviceSharding(jax.devices()[0])
        io_callback(
            self.record_fit,
            None,
            fitness,
            sharding=monitor_device,
        )

    def record_pop(self, pop, tranform=None):
        if self.record_pop_history:
            self.population_history.append(pop)
        self.current_population = pop

    def record_fit(self, fitness, metrics=None, tranform=None):
        if self.record_fit_history:
            self.fitness_history.append(fitness)

        if self.record_pf:
            if self.pf_fitness is None:
                self.pf_fitness = fitness
            else:
                self.pf_fitness = jnp.concatenate([self.pf_fitness, fitness], axis=0)

            if self.current_population is not None:
                if self.pf_solutions is None:
                    self.pf_solutions = self.current_population
                else:
                    self.pf_solutions = jnp.concatenate(
                        [self.pf_solutions, self.current_population], axis=0
                    )

            rank = non_dominated_sort(self.pf_fitness)
            pf = rank == 0
            self.pf_fitness = self.pf_fitness[pf]
            self.pf_solutions = self.pf_solutions[pf]

    def get_last(self):
        return self.opt_direction * self.fitness_history[-1]

    def get_pf_fitness(self):
        return self.opt_direction * self.pf_fitness

    def get_pf_solutions(self):
        return self.pf_solutions

    def get_history(self):
        return [self.opt_direction * fit for fit in self.fitness_history]

    def flush(self):
        hcb.barrier_wait()

    def close(self):
        self.flush()
