from typing import Any

import torch

from evox.core import Algorithm, Monitor, Problem, Workflow


class _NegModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return -x


class StdWorkflow(Workflow):
    """The standard workflow.

    ## Usage:
    ```
    algo = BasicAlgorithm(10)
    prob = BasicProblem()

    class solution_transform(nn.Module):
        def forward(self, x: torch.Tensor):
            return x / 5
    class fitness_transform(nn.Module):
        def forward(self, f: torch.Tensor):
            return -f

    monitor = EvalMonitor(full_sol_history=True)
    workflow = StdWorkflow(
        algo,
        prob,
        monitor=monitor,
        solution_transform=solution_transform(),
        fitness_transform=fitness_transform(),
    )
    monitor = workflow.get_submodule("monitor")
    workflow.init_step()
    print(monitor.topk_fitness)
    workflow.step()
    print(monitor.topk_fitness)
    # run rest of the steps ...
    ```
    """

    def __init__(self,
        algorithm: Algorithm,
        problem: Problem,
        monitor: Monitor | None = None,
        opt_direction: str = "min",
        solution_transform: torch.nn.Module | None = None,
        fitness_transform: torch.nn.Module | None = None,
        device: str | torch.device | int | None = None,
    ):
        """Initialize the standard workflow with static arguments.

        :param algorithm: The algorithm to be used in the workflow.
        :param problem: The problem to be used in the workflow.
        :param monitors: The monitors to be used in the workflow. Defaults to None. Notice: usually, monitors can only be used when using JIT script mode.
        :param opt_direction: The optimization direction, can only be "min" or "max". Defaults to "min". If "max", the fitness will be negated prior to `fitness_transform` and monitor.
        :param solution_transform: The solution transformation function. MUST be JIT-compatible module/function for JIT trace mode or a plain module for JIT script mode (default mode). Defaults to None.
        :param fitness_transforms: The fitness transformation function. MUST be JIT-compatible module/function for JIT trace mode or a plain module for JIT script mode (default mode). Defaults to None.
        :param device: The device of the workflow. Defaults to None.
        """
        super().__init__()
        assert opt_direction in [
            "min",
            "max",
        ], f"Expect optimization direction to be `min` or `max`, got {opt_direction}"
        self.opt_direction = 1 if opt_direction == "min" else -1
        if device is None:
            device = torch.get_default_device()
        # transform
        if solution_transform is None:
            solution_transform = torch.nn.Identity()
        if fitness_transform is None:
            fitness_transform = torch.nn.Identity()
        if self.opt_direction == -1:
            fitness_transform = torch.nn.Sequential(_NegModule(), fitness_transform)
        assert callable(solution_transform), f"Expect solution transform to be callable, got {solution_transform}"
        assert callable(fitness_transform), f"Expect fitness transform to be callable, got {fitness_transform}"

        if isinstance(solution_transform, torch.nn.Module):
            solution_transform.to(device=device)
        if isinstance(fitness_transform, torch.nn.Module):
            fitness_transform.to(device=device)

        self.algorithm = algorithm
        self._has_init_ = type(algorithm).init_step != Algorithm.init_step

        if monitor is None:
            monitor = Monitor()
        else:
            monitor.set_config(opt_direction=self.opt_direction)

        # set algorithm evaluate
        self.algorithm.evaluate = self._evaluate
        self.monitor = monitor
        self.problem = problem
        self.solution_transform = solution_transform
        self.fitness_transform = fitness_transform

    def get_submodule(self, target: str) -> Any:
        return super().get_submodule(target)

    def _evaluate(self, population: torch.Tensor) -> torch.Tensor:
        self.monitor.post_ask(population)
        population = self.solution_transform(population)
        self.monitor.pre_eval(population)
        fitness = self.problem.evaluate(population)
        self.monitor.post_eval(fitness)
        fitness = self.fitness_transform(fitness)
        self.monitor.pre_tell(fitness)
        return fitness

    def _step(self, init: bool):
        if init and self._has_init_:
            self.algorithm.init_step()
        else:
            self.algorithm.step()

        # If the monitor has override the `record_auxiliary` method, it will be called here.
        if "record_auxiliary" in self.monitor.__class__.__dict__:
            self.monitor.record_auxiliary(self.algorithm.record_step())

    def init_step(self):
        """
        Perform the first optimization step of the workflow.

        Calls the `init_step` of the algorithm if overwritten; otherwise, its `step` method will be invoked.
        """
        self._step(init=True)

    def step(self):
        """Perform a single optimization step using the algorithm and the problem."""
        self._step(init=False)
