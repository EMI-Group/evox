import sys
from typing import Sequence, Tuple, Callable, Any

sys.path.append(__file__ + "/../..")

import torch
from core import Monitor


class EvalMonitor(Monitor):
    """Evaluation monitor.
    Used for both single-objective and multi-objective workflow.
    Hooked around the evaluation process,
    can monitor the offspring, their corresponding fitness and keep track of the evaluation count.
    Moreover, it can also record the best solution or the pareto front on-the-fly.
    """
    def __init__(self, opt_direction: str = "min", multi_obj: bool = False, full_fit_history: bool = True, full_sol_history: bool = False, topk: int = 1):
        """Initialize the monitor.

        Args:
            opt_direction (`str`, optional): The optimization direction ("min" or "max"). Defaults to "min".
            multi_obj (`bool`, optional): Whether the optimization is multi-objective.. Defaults to False.
            full_fit_history (`bool`, optional): Whether to record the full history of fitness value. Default to True. Setting it to False may reduce memory usage.
            full_sol_history (`bool`, optional): Whether to record the full history of solutions. Default to False. Setting it to True may increase memory usage.
            topk (`int`, optional): Only affect Single-objective optimization. The number of elite solutions to record. Default to 1, which will record the best individual.
        """
        assert opt_direction in ["min", "max"]
        self.opt_direction = 1 if opt_direction == "min" else -1
        self.multi_obj = multi_obj
        self.full_fit_history = full_fit_history
        self.full_sol_history = full_sol_history
        self.topk = topk

    def set_config(self, **config):
        if "opt_direction" in config:
            self.multi_obj = config["opt_direction"]
        if "multi_obj" in config:
            self.multi_obj = config["multi_obj"]
        if "full_fit_history" in config:
            self.full_fit_history = config["full_fit_history"]
        if "full_sol_history" in config:
            self.full_sol_history = config["full_sol_history"]
        if "topk" in config:
            self.topk = config["topk"]
        return self

    def setup(self):
        self.
        return self

    def pre_step(self):
        raise NotImplementedError()

    def pre_ask(self):
        raise NotImplementedError()

    def post_ask(self, candidate_solution: torch.Tensor | Any):
        raise NotImplementedError()

    def pre_eval(self, transformed_candidate_solution: torch.Tensor | Any):
        raise NotImplementedError()

    def post_eval(self, fitness: torch.Tensor):
        raise NotImplementedError()

    def pre_tell(self, transformed_fitness: torch.Tensor):
        raise NotImplementedError()

    def post_tell(self):
        raise NotImplementedError()

    def post_step(self):
        raise NotImplementedError()