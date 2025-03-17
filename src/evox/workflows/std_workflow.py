from typing import Any, Dict

import torch

from ..core import Algorithm, Monitor, Problem, Workflow, jit_class
from ..core.module import _WrapClassBase


class _NegModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return -x


@jit_class
class StdWorkflow(Workflow):
    """The standard workflow.

    ## Usage:
    ```
    algo = BasicAlgorithm(10)
    algo.setup(-10 * torch.ones(2), 10 * torch.ones(2))
    prob = BasicProblem()

    class solution_transform(nn.Module):
        def forward(self, x: torch.Tensor):
            return x / 5
    class fitness_transform(nn.Module):
        def forward(self, f: torch.Tensor):
            return -f

    monitor = EvalMonitor(full_sol_history=True)
    workflow = StdWorkflow()
    workflow.setup(algo, prob, solution_transform=solution_transform(), fitness_transform=fitness_transform(), monitor=monitor)
    monitor = workflow.get_submodule("monitor")
    workflow.init_step()
    print(monitor.topk_fitness)
    workflow.step()
    print(monitor.topk_fitness)
    # run rest of the steps ...
    ```
    """

    def __init__(
        self,
        algorithm: Algorithm,
        problem: Problem,
        monitor: Monitor | None = None,
        solution_transform: torch.nn.Module | None = None,
        fitness_transform: torch.nn.Module | None = None,
        device: str | torch.device | int | None = None,
        algorithm_setup_params: Dict[str, Any] | None = None,
        problem_setup_params: Dict[str, Any] | None = None,
        monitor_setup_params: Dict[str, Any] | None = None,
    ):
        """Setup the module with submodule initialization. Since all of these arguments are mutable modules to be added as submodules, they are placed here instead of `__init__` and thus `setup` MUST be invoked after `__init__`.

        :param algorithm: The algorithm to be used in the workflow.
        :param problem: The problem to be used in the workflow.
        :param monitors: The monitors to be used in the workflow. Defaults to None.
        :param opt_direction: The optimization direction, can only be "min" or "max". Defaults to "min". If "max", the fitness will be negated prior to `fitness_transform` and monitor.
        :param solution_transform: The solution transformation function. MUST be compile-compatible module/function. Defaults to None.
        :param fitness_transforms: The fitness transformation function. MUST be compile-compatible module/function. Defaults to None.
        :param device: The device of the workflow. Defaults to None.

        :note: The `algorithm`, `problem`, `solution_transform`, and `fitness_transform` will be IN-PLACE moved to the device specified by `device`.
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
        elif isinstance(solution_transform, _WrapClassBase):  # ensure correct results for jit_class
            solution_transform = solution_transform.__inner_module__
        if fitness_transform is None:
            fitness_transform = torch.nn.Identity()
        elif isinstance(fitness_transform, _WrapClassBase):  # ensure correct results for jit_class
            fitness_transform = fitness_transform.__inner_module__
        if self.opt_direction == -1:
            fitness_transform = torch.nn.Sequential(_NegModule(), fitness_transform)
        assert callable(solution_transform), f"Expect solution transform to be callable, got {solution_transform}"
        assert callable(fitness_transform), f"Expect fitness transform to be callable, got {fitness_transform}"
        if isinstance(solution_transform, torch.nn.Module):
            solution_transform.to(device=device)
        if isinstance(fitness_transform, torch.nn.Module):
            fitness_transform.to(device=device)

        if monitor is None:
            monitor = Monitor()
        else:
            monitor.set_config(opt_direction=self.opt_direction)
        algorithm.to(device=device)
        monitor.to(device=device)
        problem.to(device=device)

        # set algorithm evaluate
        self._has_init_ = type(algorithm).init_step != Algorithm.init_step

        class _SubAlgorithm(type(algorithm)):
            def __init__(self_algo):
                super(type(algorithm), self_algo).__init__()
                self_algo.__dict__.update(algorithm.__dict__)

            def evaluate(self_algo, pop: torch.Tensor) -> torch.Tensor:
                return self._evaluate(pop)

        # set submodules
        self.algorithm = _SubAlgorithm()
        self.monitor = monitor
        self.problem = problem
        self.solution_transform = solution_transform
        self.fitness_transform = fitness_transform

    def __getattribute__(self, name: str):
        if name == "_monitor_":
            return self.algorithm._monitor_
        elif name == "_problem_":
            return self.algorithm._problem_
        elif name == "_solution_transform_":
            return self.algorithm._solution_transform_
        elif name == "_fitness_transform_":
            return self.algorithm._fitness_transform_
        return super().__getattribute__(name)

    def __sync_with__(self, jit_module):
        if "_monitor_" in self._modules:
            del self._monitor_
            del self._problem_
            del self._fitness_transform_
            del self._solution_transform_
        return super().__sync_with__(jit_module)

    @torch.jit.ignore
    def get_submodule(self, target: str):
        if target == "monitor":
            return self.algorithm._monitor_
        elif target == "problem":
            return self.algorithm._problem_
        return super().get_submodule(target)

    def _evaluate(self, population: torch.Tensor) -> torch.Tensor:
        self._monitor_.post_ask(population)
        population = self._solution_transform_(population)
        self._monitor_.pre_eval(population)
        fitness = self._problem_.evaluate(population)
        self._monitor_.post_eval(fitness)
        fitness = self._fitness_transform_(fitness)
        self._monitor_.pre_tell(fitness)
        return fitness

    def _step(self, init: bool):
        if init and self._has_init_:
            self.algorithm.init_step()
        else:
            self.algorithm.step()

        self.algorithm.record_step()

    def init_step(self):
        """
        Perform the first optimization step of the workflow.

        Calls the `init_step` of the algorithm if overwritten; otherwise, its `step` method will be invoked.
        """
        self._step(init=True)

    def step(self):
        """Perform a single optimization step using the algorithm and the problem."""
        self._step(init=False)
