import warnings

import jax
import jax.numpy as jnp

from evox import Monitor, dataclass, pytree_field
from evox.vis_tools import plot


@dataclass
class EvalMonitorState:
    first_step: bool = pytree_field(static=True)
    latest_solution: jax.Array
    latest_fitness: jax.Array
    topk_solutions: jax.Array
    topk_fitness: jax.Array


class EvalMonitor(Monitor):
    """Evaluation monitor.
    Used for both single-objective and multi-objective workflow.
    Hooked around the evaluation process,
    can monitor the offspring, their corresponding fitness and keep track of the evaluation count.
    Moreover, it can also record the best solution or the pareto front on-the-fly.

    Parameters
    ----------
    multi_obj
        Whether the optimization is multi-objective.
        Default to False.
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
    parallel_monitor
        Enable the use of parallel monitor,
        that is monitoring multiple optimization process in parallel.
        Typically used in meta-optimization settings where the use of vmap of a workflow is needed.
    """

    def __init__(
        self,
        multi_obj=False,
        full_fit_history=True,
        full_sol_history=False,
        topk=1,
    ):
        self.multi_obj = multi_obj
        self.full_fit_history = full_fit_history
        self.full_sol_history = full_sol_history
        self.topk = topk
        self.fitness_history = []
        self.solution_history = []
        self.opt_direction = 1  # default to min, so no transformation is needed

    def hooks(self):
        return ["post_ask", "post_eval"]

    def set_opt_direction(self, opt_direction):
        self.opt_direction = opt_direction

    def setup(self, _key):
        return EvalMonitorState(
            first_step=True,
            latest_solution=None,
            latest_fitness=None,
            topk_solutions=None,
            topk_fitness=None,
        )

    def post_ask(self, state, _workflow_state, candidate):
        return state.replace(latest_solution=candidate)

    def post_eval(self, state, _workflow_state, fitness):
        if fitness.ndim == 1:
            # single-objective
            if state.first_step:
                topk_solutions = state.latest_solution
                topk_fitness = fitness
                state = state.replace(first_step=False)
            else:
                topk_solutions = jnp.concatenate(
                    [state.topk_solutions, state.latest_solution]
                )
                topk_fitness = jnp.concatenate([state.topk_fitness, fitness])
            rank = jnp.argsort(topk_fitness)
            topk_solutions = topk_solutions[rank[: self.topk]]
            topk_fitness = topk_fitness[rank[: self.topk]]
            state = state.replace(
                topk_solutions=topk_solutions,
                topk_fitness=topk_fitness,
                latest_fitness=fitness,
            )
        else:
            # multi-objective
            state = state.replace(latest_fitness=fitness)

        if self.full_fit_history or self.full_sol_history:
            return state.register_callback(
                self._record_history,
                state.latest_solution if self.full_sol_history else None,
                fitness if self.full_fit_history else None,
            )
        else:
            return state

    def _record_history(self, solution, fitness):
        # since history is a list, which doesn't have a static shape
        # we need to use register_callback to record the history
        if self.full_sol_history:
            self.solution_history.append(solution)
        if self.full_fit_history:
            self.fitness_history.append(fitness)

    def get_latest_fitness(self, state):
        return self.opt_direction * state.latest_fitness, state

    def get_latest_solution(self, state):
        return state.latest_solution, state

    def get_topk_fitness(self, state):
        return self.opt_direction * state.topk_fitness, state

    def get_topk_solutions(self, state):
        return state.topk_solutions, state

    def get_best_solution(self, state):
        return state.topk_solutions[0], state

    def get_best_fitness(self, state):
        return self.opt_direction * state.topk_fitness[0], state

    def get_fitness_history(self, state):
        return [self.opt_direction * fit for fit in self.fitness_history], state

    def get_solution_history(self, state):
        return self.solution_history, state

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
