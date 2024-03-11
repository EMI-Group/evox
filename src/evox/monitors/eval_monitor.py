import jax
import jax.experimental.host_callback as hcb
import jax.numpy as jnp
import numpy as np
from jax.experimental import io_callback
from jax.sharding import SingleDeviceSharding

from evox import Monitor

from ..operators import non_dominated_sort


class EvalMonitor(Monitor):
    """Evaluation monitor.
    Used for both single-objective and multi-objective workflow.
    Hooked around the evaluation process,
    can monitor the offspring, their corresponding fitness and keep track of the evaluation count.
    Moreover, it can also record the best solution or the pareto front on-the-fly.

    Parameters
    ----------
    full_fit_history
        Whether to record the full history of fitness value.
        Default to True. Setting it to False may reduce memory usage.
    full_sol_history
        Whether to record the full history of solutions.
        Default to False. Setting it to True may increase memory usage.
    topk
        Only affect Single-objective optimization. The number of elite solutions to record.
        Default to 1, which will record the best individual.
    calc_pf
        Only affect Multi-objective optimization.
        Whether to keep updating the pareto front during the run. (The Archive)
        Default to False.
        Setting it to True will cause the monitor to
        maintain a pareto front of all the solutions with unlimited size,
        which may hurt performance.
    """

    def __init__(
        self, full_fit_history=True, full_sol_history=False, topk=1, calc_pf=False
    ):
        self.full_fit_history = full_fit_history
        self.full_sol_history = full_sol_history
        self.topk = topk
        self.calc_pf = calc_pf
        self.fitness_history = []
        self.solution_history = []
        self.current_solutions = None
        self.pf_solutions = None
        self.pf_fitness = None
        self.eval_count = 0
        self.opt_direction = 1  # default to min, so no transformation is needed

    def hooks(self):
        return ["post_ask", "post_eval"]

    def set_opt_direction(self, opt_direction):
        self.opt_direction = opt_direction

    def post_ask(self, _state, cand_sol):
        monitor_device = SingleDeviceSharding(jax.devices()[0])
        io_callback(self.record_sol, None, cand_sol, sharding=monitor_device)

    def post_eval(self, _state, _cand_sol, _transformed_cand_sol, fitness):
        monitor_device = SingleDeviceSharding(jax.devices()[0])
        if fitness.ndim == 1:
            recorder = self.record_fit_single_obj
        else:
            recorder = self.record_fit_multi_obj

        io_callback(
            recorder,
            None,
            fitness,
            sharding=monitor_device,
        )

    def record_sol(self, sols):
        if self.full_sol_history:
            self.solution_history.append(sols)
        self.current_solutions = sols

    def record_fit_single_obj(self, fitness):
        if self.full_fit_history:
            self.fitness_history.append(fitness)
        if self.topk == 1:
            # handle the case where topk = 1
            # don't need argsort / top_k, which are slower
            current_min_fit = jnp.min(fitness, keepdims=True)
            if self.topk_fitness is None or self.topk_fitness > current_min_fit:
                self.topk_fitness = current_min_fit
                if self.current_solutions is not None:
                    individual_index = jnp.argmin(fitness)
                    # use slice to keepdim,
                    # because topk_solutions should have dim of (1, dim)
                    self.topk_solutions = self.current_solutions[
                        individual_index : individual_index + 1
                    ]
        else:
            # since topk > 1, we have to sort the fitness
            if self.topk_fitness is None:
                self.topk_fitness = fitness
            else:
                self.topk_fitness = jnp.concatenate([self.topk_fitness, fitness])

            if self.current_solutions is not None:
                if self.topk_solutions is None:
                    self.topk_solutions = self.current_solutions
                else:
                    self.topk_solutions = jnp.concatenate(
                        [self.topk_solutions, self.current_solutions], axis=0
                    )
            rank = jnp.argsort(self.topk_fitness)
            topk_rank = rank[: self.topk]
            if self.current_solutions is not None:
                self.topk_solutions = self.topk_solutions[topk_rank]
            self.topk_fitness = self.topk_fitness[topk_rank]

    def record_fit_multi_obj(self, fitness):
        if self.full_fit_history:
            self.fitness_history.append(fitness)

        if self.calc_pf:
            if self.pf_fitness is None:
                self.pf_fitness = fitness
            else:
                self.pf_fitness = jnp.concatenate([self.pf_fitness, fitness], axis=0)

            if self.current_solutions is not None:
                if self.pf_solutions is None:
                    self.pf_solutions = self.current_solutions
                else:
                    self.pf_solutions = jnp.concatenate(
                        [self.pf_solutions, self.current_solutions], axis=0
                    )

            rank = non_dominated_sort(self.pf_fitness)
            pf = rank == 0
            self.pf_fitness = self.pf_fitness[pf]
            self.pf_solutions = self.pf_solutions[pf]

    def get_latest_fitness(self):
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
