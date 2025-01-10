from typing import List

import torch
from torch import nn

from ..core import Monitor


class EvalMonitor(Monitor):
    """Evaluation monitor.
    Used for both single-objective and multi-objective workflow.
    Hooked around the evaluation process,
    can monitor the offspring, their corresponding fitness and keep track of the evaluation count.
    Moreover, it can also record the best solution or the pareto front on-the-fly.
    """

    def __init__(
        self,
        multi_obj: bool = False,
        full_fit_history: bool = True,
        full_sol_history: bool = False,
        topk: int = 1,
        device: torch.device | None = None,
    ):
        """Initialize the monitor.

        :param multi_obj: Whether the optimization is multi-objective. Defaults to False.
        :param full_fit_history: Whether to record the full history of fitness value. Default to True. Setting it to False may reduce memory usage.
        :param full_sol_history: Whether to record the full history of solutions. Default to False. Setting it to True may increase memory usage.
        :param topk: Only affect Single-objective optimization. The number of elite solutions to record. Default to 1, which will record the best individual.
        :param device: The device of the monitor. Defaults to None.
        """
        super().__init__()
        self.multi_obj = multi_obj
        self.full_fit_history = full_fit_history
        self.full_sol_history = full_sol_history
        self.topk = topk
        # mutable
        self.latest_solution = nn.Buffer(torch.empty(0, device=device))
        self.latest_fitness = nn.Buffer(torch.empty(0, device=device))
        self.topk_solutions = nn.Buffer(torch.empty(0, device=device))
        self.topk_fitness = nn.Buffer(torch.empty(0, device=device))
        self.fitness_history: List[torch.Tensor] = [torch.empty(0, device=device)]
        self.solution_history: List[torch.Tensor] = [torch.empty(0, device=device)]

    def set_config(self, **config):
        if "multi_obj" in config:
            self.multi_obj = config["multi_obj"]
        if "full_fit_history" in config:
            self.full_fit_history = config["full_fit_history"]
        if "full_sol_history" in config:
            self.full_sol_history = config["full_sol_history"]
        if "topk" in config:
            self.topk = config["topk"]
        return self

    def post_ask(self, candidate_solution: torch.Tensor):
        self.latest_solution = candidate_solution

    def pre_tell(self, fitness: torch.Tensor):
        self.latest_fitness = fitness
        if fitness.ndim == 1:
            # single-objective
            self.multi_obj = False
            assert fitness.size(0) >= self.topk
            if self.topk_solutions.ndim <= 1:
                topk_solutions = self.latest_solution
                topk_fitness = fitness
                rank = torch.argsort(topk_fitness)[: self.topk]
                self.topk_fitness = topk_fitness[rank]
                self.topk_solutions = topk_solutions[rank]
            else:
                topk_solutions = torch.concatenate([self.topk_solutions, self.latest_solution])
                topk_fitness = torch.concatenate([self.topk_fitness, fitness])
                rank = torch.argsort(topk_fitness)[: self.topk]
                self.topk_fitness.copy_(topk_fitness[rank])
                self.topk_solutions.copy_(topk_solutions[rank])
        elif fitness.ndim == 2:
            # multi-objective
            self.multi_obj = True
        else:
            raise ValueError(f"Invalid fitness shape: {fitness.shape}")

        if self.full_fit_history or self.full_sol_history:
            if self.full_sol_history:
                self.solution_history.append(self.latest_solution)
            if self.full_fit_history:
                self.fitness_history.append(self.latest_fitness)

    # TODO: modify the commented methods to fit this framework
    # def get_latest_fitness(self, state) -> Tuple[jax.Array, EvalMonitorState]:
    #     """Get the fitness values from the latest iteration."""
    #     return self.opt_direction * state.latest_fitness, state

    # def get_latest_solution(self, state) -> Tuple[jax.Array, EvalMonitorState]:
    #     """Get the solution from the latest iteration."""
    #     return state.latest_solution, state

    # def get_topk_fitness(self, state) -> Tuple[jax.Array, EvalMonitorState]:
    #     """Get the topk fitness values so far."""
    #     return self.opt_direction * state.topk_fitness, state

    # def get_topk_solutions(self, state) -> Tuple[jax.Array, EvalMonitorState]:
    #     """Get the topk solutions so far."""
    #     return state.topk_solutions, state

    # def get_best_solution(self, state) -> Tuple[jax.Array, EvalMonitorState]:
    #     """Get the best solution so far."""
    #     return state.topk_solutions[..., 0, :], state

    # def get_best_fitness(self, state) -> Tuple[jax.Array, EvalMonitorState]:
    #     """Get the best fitness value so far."""
    #     if self.multi_obj:
    #         raise ValueError(
    #             "Multi-objective optimization does not have a single best fitness."
    #         )
    #     return self.opt_direction * state.topk_fitness[..., 0], state

    # def get_fitness_history(
    #     self, state=None
    # ) -> Tuple[list[jax.Array], EvalMonitorState]:
    #     """Get the full history of fitness values."""
    #     return [self.opt_direction * fit for fit in self.fitness_history], state

    # def get_solution_history(
    #     self, state=None
    # ) -> Tuple[list[jax.Array], EvalMonitorState]:
    #     """Get the full history of solutions."""
    #     return self.solution_history, state

    # def get_pf_fitness(self, state=None) -> Tuple[jax.Array, EvalMonitorState]:
    #     """Get the approximate pareto front fitness values of all the solutions evaluated so far.
    #     Requires enabling `full_fit_history`."""
    #     if not self.full_fit_history:
    #         warnings.warn("`get_pf_fitness` requires enabling `full_fit_history`.")
    #     all_fitness, _ = self.get_fitness_history(state)
    #     all_fitness = jnp.concatenate(all_fitness, axis=0)
    #     all_fitness = jnp.unique(all_fitness, axis=0)
    #     rank = non_dominated_sort(all_fitness)
    #     return all_fitness[rank == 0], state

    # def get_pf_solutions(
    #     self, state=None, deduplicate=True
    # ) -> Tuple[jax.Array, EvalMonitorState]:
    #     """Get the approximate pareto front solutions of all the solutions evaluated so far.
    #     Requires enabling both `full_sol_history` and `full_sol_history`.
    #     If `deduplicate` is set to True, the duplicated solutions will be removed."""
    #     pf_solutions, _pf_fitness, _ = self.get_pf(state, deduplicate)
    #     return pf_solutions, state

    # def get_pf(
    #     self, state=None, deduplicate=True
    # ) -> Tuple[jax.Array, jax.Array, EvalMonitorState]:
    #     """Get the approximate pareto front solutions and fitness values of all the solutions evaluated so far.
    #     Requires enabling both `full_sol_history` and `full_sol_history`.
    #     If `deduplicate` is set to True, the duplicated solutions will be removed."""
    #     if not self.full_fit_history or not self.full_sol_history:
    #         warnings.warn(
    #             "`get_pf` requires enabling both `full_sol_history` and `full_sol_history`."
    #         )
    #     all_solutions, _ = self.get_solution_history(state)
    #     all_solutions = jnp.concatenate(all_solutions, axis=0)
    #     all_fitness, _ = self.get_fitness_history(state)
    #     all_fitness = jnp.concatenate(all_fitness, axis=0)

    #     if deduplicate:
    #         _, unique_index = jnp.unique(
    #             all_solutions,
    #             axis=0,
    #             return_index=True,
    #         )
    #         all_solutions = all_solutions[unique_index]
    #         all_fitness = all_fitness[unique_index]

    #     rank = non_dominated_sort(all_fitness)
    #     pf_fitness = all_fitness[rank == 0]
    #     pf_solutions = all_solutions[rank == 0]
    #     return pf_solutions, pf_fitness, state

    # def plot(self, state=None, problem_pf=None, **kwargs):
    #     if not self.fitness_history:
    #         warnings.warn("No fitness history recorded, return None")
    #         return

    #     if self.fitness_history[0].ndim == 1:
    #         n_objs = 1
    #     else:
    #         n_objs = self.fitness_history[0].shape[1]

    #     fitness_history, _ = self.get_fitness_history(state)

    #     if n_objs == 1:
    #         return plot.plot_obj_space_1d(fitness_history, **kwargs)
    #     elif n_objs == 2:
    #         return plot.plot_obj_space_2d(fitness_history, problem_pf, **kwargs)
    #     elif n_objs == 3:
    #         return plot.plot_obj_space_3d(fitness_history, problem_pf, **kwargs)
    #     else:
    #         warnings.warn("Not supported yet.")
