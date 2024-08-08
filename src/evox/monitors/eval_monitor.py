import warnings

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import io_callback
from jax.sharding import SingleDeviceSharding

from evox import Monitor
from evox.vis_tools import plot

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
        Default to False.
        Setting it to True may increase memory usage,
        and adding a significant overhead of transfering the entire solutions set from GPU to CPU.
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
        self.topk_fitness = None
        self.topk_solutions = None
        self.pf_solutions = None
        self.pf_fitness = None
        self.latest_solution = None
        self.latest_fitness = None
        self.eval_count = 0
        self.opt_direction = 1  # default to min, so no transformation is needed

    def hooks(self):
        return ["post_eval"]

    def set_opt_direction(self, opt_direction):
        self.opt_direction = opt_direction

    def post_eval(self, state, cand_sol, _transformed_cand_sol, fitness):
        if fitness.ndim == 1:
            if self.full_sol_history:
                cand_fit = None
            else:
                # when not recording full solution history,
                # only send the topk solutions to the host to save bandwidth
                rank = jnp.argsort(fitness)
                topk_rank = rank[: self.topk]
                cand_sol = cand_sol[topk_rank]
                cand_fit = fitness[topk_rank]

            return state.register_callback(
                self.record_fit_single_obj,
                cand_sol,
                cand_fit,
                fitness,
            )
        else:
            return state.register_callback(
                self.record_fit_multi_obj,
                cand_sol,
                fitness,
            )

    def record_fit_single_obj(self, cand_sol, cand_fit, fitness):
        if cand_fit is None:
            cand_fit = fitness

        if self.full_sol_history:
            self.solution_history.append(cand_sol)

        if self.full_fit_history:
            self.fitness_history.append(fitness)

        if self.topk == 1:
            # handle the case where topk = 1
            # don't need argsort / top_k, which are slower
            current_min_fit = jnp.min(cand_fit, keepdims=True)
            if self.topk_fitness is None or self.topk_fitness > current_min_fit:
                self.topk_fitness = current_min_fit
                individual_index = jnp.argmin(cand_fit)
                # use slice to keepdim,
                # because topk_solutions should have dim of (1, dim)
                self.topk_solutions = cand_sol[individual_index : individual_index + 1]
        else:
            # since topk > 1, we have to sort the fitness
            if self.topk_fitness is None:
                self.topk_fitness = cand_fit
            else:
                self.topk_fitness = jnp.concatenate([self.topk_fitness, cand_fit])

            if self.topk_solutions is None:
                self.topk_solutions = cand_sol
            else:
                self.topk_solutions = jnp.concatenate(
                    [self.topk_solutions, cand_sol], axis=0
                )
            rank = jnp.argsort(self.topk_fitness)
            topk_rank = rank[: self.topk]
            self.topk_solutions = self.topk_solutions[topk_rank]
            self.topk_fitness = self.topk_fitness[topk_rank]

    def record_fit_multi_obj(self, cand_sol, fitness):
        if self.full_sol_history:
            self.solution_history.append(cand_sol)

        if self.full_fit_history:
            self.fitness_history.append(fitness)

        if self.calc_pf:
            if self.pf_fitness is None:
                self.pf_fitness = fitness
            else:
                self.pf_fitness = jnp.concatenate([self.pf_fitness, fitness], axis=0)

            if self.pf_solutions is None:
                self.pf_solutions = cand_sol
            else:
                self.pf_solutions = jnp.concatenate(
                    [self.pf_solutions, cand_sol], axis=0
                )

            rank = non_dominated_sort(self.pf_fitness)
            pf = rank == 0
            self.pf_fitness = self.pf_fitness[pf]
            self.pf_solutions = self.pf_solutions[pf]

        self.latest_fitness = fitness
        self.latest_solution = cand_sol

    def get_latest_fitness(self):
        return self.opt_direction * self.latest_fitness

    def get_latest_solution(self):
        return self.latest_solution

    def get_pf_fitness(self):
        return self.opt_direction * self.pf_fitness

    def get_pf_solutions(self):
        return self.pf_solutions

    def get_topk_fitness(self):
        return self.opt_direction * self.topk_fitness

    def get_topk_solutions(self):
        return self.topk_solutions

    def get_best_solution(self):
        return self.topk_solutions[0]

    def get_best_fitness(self):
        return self.opt_direction * self.topk_fitness[0]

    def get_history(self):
        return [self.opt_direction * fit for fit in self.fitness_history]

    def plot(self, problem_pf=None, **kwargs):
        if not self.fitness_history:
            warnings.warn("No fitness history recorded, return None")
            return

        if self.fitness_history[0].ndim == 1:
            n_objs = 1
        else:
            n_objs = self.fitness_history[0].shape[1]

        if n_objs == 1:
            return plot.plot_obj_space_1d(self.get_history(), **kwargs)
        elif n_objs == 2:
            return plot.plot_obj_space_2d(self.get_history(), problem_pf, **kwargs)
        elif n_objs == 3:
            return plot.plot_obj_space_3d(self.get_history(), problem_pf, **kwargs)
        else:
            warnings.warn("Not supported yet.")

    def flush(self):
        jax.effects_barrier()

    def close(self):
        self.flush()
