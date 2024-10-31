from typing import Tuple
import warnings

import jax
import jax.numpy as jnp

from evox import Monitor, dataclass, pytree_field
from evox.vis_tools import plot
from evox.operators import non_dominated_sort


@dataclass
class EvalMonitorState:
    first_step: bool = pytree_field(static=True)
    latest_solution: jax.Array
    latest_fitness: jax.Array
    topk_solutions: jax.Array
    topk_fitness: jax.Array


@dataclass
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
    """

    multi_obj: bool = pytree_field(default=False, static=True)
    full_fit_history: bool = pytree_field(default=True, static=True)
    full_sol_history: bool = pytree_field(default=False, static=True)
    topk: int = pytree_field(default=1, static=True)

    opt_direction: int = pytree_field(static=True, init=False, default=1)
    fitness_history: list = pytree_field(static=True, init=False, default_factory=list)
    solution_history: list = pytree_field(static=True, init=False, default_factory=list)

    def hooks(self):
        return ["post_ask", "post_eval"]

    def set_opt_direction(self, opt_direction):
        self.set_frozen_attr("opt_direction", opt_direction)

    def setup(self, _key):
        return EvalMonitorState(
            first_step=True,
            latest_solution=None,
            latest_fitness=None,
            topk_solutions=None,
            topk_fitness=None,
        )

    def clear_history(self):
        """Clear the history of fitness and solutions.
        Normally it will be called at the initialization of Workflow object."""
        self.fitness_history = []
        self.solution_history = []

    def post_ask(self, state, _workflow_state, candidate):
        return state.replace(latest_solution=candidate)

    def post_eval(self, state, _workflow_state, fitness):
        if fitness.ndim == 1:
            # single-objective
            self.multi_obj = False
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
            self.multi_obj = True
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

    def get_latest_fitness(self, state) -> Tuple[jax.Array, EvalMonitorState]:
        """Get the fitness values from the latest iteration."""
        return self.opt_direction * state.latest_fitness, state

    def get_latest_solution(self, state) -> Tuple[jax.Array, EvalMonitorState]:
        """Get the solution from the latest iteration."""
        return state.latest_solution, state

    def get_topk_fitness(self, state) -> Tuple[jax.Array, EvalMonitorState]:
        """Get the topk fitness values so far."""
        return self.opt_direction * state.topk_fitness, state

    def get_topk_solutions(self, state) -> Tuple[jax.Array, EvalMonitorState]:
        """Get the topk solutions so far."""
        return state.topk_solutions, state

    def get_best_solution(self, state) -> Tuple[jax.Array, EvalMonitorState]:
        """Get the best solution so far."""
        return state.topk_solutions[..., 0, :], state

    def get_best_fitness(self, state) -> Tuple[jax.Array, EvalMonitorState]:
        """Get the best fitness value so far."""
        if self.multi_obj:
            raise ValueError(
                "Multi-objective optimization does not have a single best fitness."
            )
        return self.opt_direction * state.topk_fitness[..., 0], state

    def get_fitness_history(
        self, state=None
    ) -> Tuple[list[jax.Array], EvalMonitorState]:
        """Get the full history of fitness values."""
        return [self.opt_direction * fit for fit in self.fitness_history], state

    def get_solution_history(
        self, state=None
    ) -> Tuple[list[jax.Array], EvalMonitorState]:
        """Get the full history of solutions."""
        return self.solution_history, state

    def get_pf_fitness(self, state=None) -> Tuple[jax.Array, EvalMonitorState]:
        """Get the approximate pareto front fitness values of all the solutions evaluated so far.
        Requires enabling `full_fit_history`."""
        if not self.full_fit_history:
            warnings.warn("`get_pf_fitness` requires enabling `full_fit_history`.")
        all_fitness, _ = self.get_fitness_history(state)
        all_fitness = jnp.concatenate(all_fitness, axis=0)
        all_fitness = jnp.unique(all_fitness, axis=0)
        rank = non_dominated_sort(all_fitness)
        return all_fitness[rank == 0], state

    def get_pf_solutions(
        self, state=None, deduplicate=True
    ) -> Tuple[jax.Array, EvalMonitorState]:
        """Get the approximate pareto front solutions of all the solutions evaluated so far.
        Requires enabling both `full_sol_history` and `full_sol_history`.
        If `deduplicate` is set to True, the duplicated solutions will be removed."""
        pf_solutions, _pf_fitness, _ = self.get_pf(state, deduplicate)
        return pf_solutions, state

    def get_pf(
        self, state=None, deduplicate=True
    ) -> Tuple[jax.Array, jax.Array, EvalMonitorState]:
        """Get the approximate pareto front solutions and fitness values of all the solutions evaluated so far.
        Requires enabling both `full_sol_history` and `full_sol_history`.
        If `deduplicate` is set to True, the duplicated solutions will be removed."""
        if not self.full_fit_history or not self.full_sol_history:
            warnings.warn(
                "`get_pf` requires enabling both `full_sol_history` and `full_sol_history`."
            )
        all_solutions, _ = self.get_solution_history(state)
        all_solutions = jnp.concatenate(all_solutions, axis=0)
        all_fitness, _ = self.get_fitness_history(state)
        all_fitness = jnp.concatenate(all_fitness, axis=0)

        if deduplicate:
            _, unique_index = jnp.unique(
                all_solutions,
                axis=0,
                return_index=True,
            )
            all_solutions = all_solutions[unique_index]
            all_fitness = all_fitness[unique_index]

        rank = non_dominated_sort(all_fitness)
        pf_fitness = all_fitness[rank == 0]
        pf_solutions = all_solutions[rank == 0]
        return pf_solutions, pf_fitness, state

    def plot(self, state=None, problem_pf=None, **kwargs):
        if not self.fitness_history:
            warnings.warn("No fitness history recorded, return None")
            return

        if self.fitness_history[0].ndim == 1:
            n_objs = 1
        else:
            n_objs = self.fitness_history[0].shape[1]

        fitness_history, _ = self.get_fitness_history(state)

        if n_objs == 1:
            return plot.plot_obj_space_1d(fitness_history, **kwargs)
        elif n_objs == 2:
            return plot.plot_obj_space_2d(fitness_history, problem_pf, **kwargs)
        elif n_objs == 3:
            return plot.plot_obj_space_3d(fitness_history, problem_pf, **kwargs)
        else:
            warnings.warn("Not supported yet.")
