import warnings
from typing import List

import torch

from ..core import Monitor, Mutable

try:
    from ..vis_tools import plot
except ImportError:
    plot = None


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
        self.device = device
        # mutable
        self.latest_solution = Mutable(torch.empty(0, device=device))
        self.latest_fitness = Mutable(torch.empty(0, device=device))
        self.topk_solutions = Mutable(torch.empty(0, device=device))
        self.topk_fitness = Mutable(torch.empty(0, device=device))
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
                rank = torch.topk(topk_fitness, self.topk)[1]
                self.topk_fitness = topk_fitness[rank]
                self.topk_solutions = topk_solutions[rank]
            else:
                topk_solutions = torch.concatenate([self.topk_solutions, self.latest_solution])
                topk_fitness = torch.concatenate([self.topk_fitness, fitness])
                rank = torch.topk(topk_fitness, self.topk)[1]
                self.topk_fitness.copy_(topk_fitness[rank])
                self.topk_solutions.copy_(topk_solutions[rank])
        elif fitness.ndim == 2:
            # multi-objective
            self.multi_obj = True
        else:
            raise ValueError(f"Invalid fitness shape: {fitness.shape}")

        if self.full_fit_history or self.full_sol_history:
            if self.full_sol_history:
                self.solution_history.append(self.latest_solution.to(self.device))
            if self.full_fit_history:
                self.fitness_history.append(self.latest_fitness.to(self.device))

    @torch.jit.ignore
    def plot(self, state=None, problem_pf=None, **kwargs):
        if not self.fitness_history:
            warnings.warn("No fitness history recorded, return None")
            return

        if plot is None:
            warnings.warn("No visualization tool available, return None")
            return

        if self.fitness_history[0].ndim == 1:
            n_objs = 1
        else:
            n_objs = self.fitness_history[0].shape[1]

        fitness_history = [f.numpy() for f in self.fitness_history[1:]]

        if n_objs == 1:
            return plot.plot_obj_space_1d(fitness_history, **kwargs)
        elif n_objs == 2:
            return plot.plot_obj_space_2d(fitness_history, problem_pf, **kwargs)
        elif n_objs == 3:
            return plot.plot_obj_space_3d(fitness_history, problem_pf, **kwargs)
        else:
            warnings.warn("Not supported yet.")
