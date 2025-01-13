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
        device = torch.get_default_device() if device is None else device
        self.multi_obj = multi_obj
        self.full_fit_history = full_fit_history
        self.full_sol_history = full_sol_history
        self.opt_direction = 1
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
        if "opt_direction" in config:
            self.opt_direction = config["opt_direction"]
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
                rank = torch.topk(topk_fitness, self.topk, largest=False)[1]
                self.topk_fitness = topk_fitness[rank]
                self.topk_solutions = topk_solutions[rank]
            else:
                topk_solutions = torch.concatenate([self.topk_solutions, self.latest_solution])
                topk_fitness = torch.concatenate([self.topk_fitness, fitness])
                rank = torch.topk(topk_fitness, self.topk, largest=False)[1]
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
    def get_latest_fitness(self) -> torch.Tensor:
        """Get the fitness values from the latest iteration."""
        return self.opt_direction * self.latest_fitness

    @torch.jit.ignore
    def get_latest_solution(self) -> torch.Tensor:
        """Get the solution from the latest iteration."""
        return self.latest_solution

    @torch.jit.ignore
    def get_topk_fitness(self) -> torch.Tensor:
        """Get the topk fitness values so far."""
        return self.opt_direction * self.topk_fitness

    @torch.jit.ignore
    def get_topk_solutions(self) -> torch.Tensor:
        """Get the topk solutions so far."""
        return self.topk_solutions

    @torch.jit.ignore
    def get_best_solution(self) -> torch.Tensor:
        """Get the best solution so far."""
        return self.topk_solutions[0]

    @torch.jit.ignore
    def get_best_fitness(self) -> torch.Tensor:
        """Get the best fitness value so far."""
        if self.multi_obj:
            raise ValueError("Multi-objective optimization does not have a single best fitness.")
        return self.opt_direction * self.topk_fitness[0]

    @torch.jit.ignore
    def get_fitness_history(self) -> List[torch.Tensor]:
        """Get the full history of fitness values."""
        return [self.opt_direction * fit for fit in self.fitness_history[1:]]

    @torch.jit.ignore
    def get_solution_history(self) -> List[torch.Tensor]:
        """Get the full history of solutions."""
        return self.solution_history[1:]

    @torch.jit.ignore
    def plot(self, problem_pf=None, **kwargs):
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

        fitness_history = self.get_fitness_history()
        fitness_history = [f.cpu().numpy() for f in fitness_history]

        if n_objs == 1:
            return plot.plot_obj_space_1d(fitness_history, **kwargs)
        elif n_objs == 2:
            return plot.plot_obj_space_2d(fitness_history, problem_pf, **kwargs)
        elif n_objs == 3:
            return plot.plot_obj_space_3d(fitness_history, problem_pf, **kwargs)
        else:
            warnings.warn("Not supported yet.")
