import torch
from ..core import Algorithm, Problem, Workflow, Monitor, jit_class


@jit_class
class StdWorkflow(Workflow):
    """The standard workflow"""

    def __init__(
        self,
        opt_direction: str = "min",
        solution_transform: torch.nn.Module | None = None,
        fitness_transform: torch.nn.Module | None = None,
    ):
        """Initialize the standard workflow with static arguments.

        Args:
            opt_direction (`str`, optional): The optimization direction, can only be "min" or "max". Defaults to "min". If "max", the fitness will be negated prior to `fitness_transform` and monitor.
            solution_transform (a `torch.nn.Module` whose forward function signature is `Callable[[torch.Tensor], torch.Tensor | Any]`, optional): The solution transformation function. MUST be JIT-compatible module/function for JIT trace mode or a plain module for JIT script mode (default mode). Defaults to None.
            fitness_transforms (a `torch.nn.Module` whose forward function signature is `Callable[[torch.Tensor], torch.Tensor]`, optional): The fitness transformation function. MUST be JIT-compatible module/function for JIT trace mode or a plain module for JIT script mode (default mode). Defaults to None.
        """
        super().__init__()
        assert opt_direction in [
            "min",
            "max",
        ], f"Expect optimization direction to be `min` or `max`, got {opt_direction}"
        self.opt_direction = 1 if opt_direction == "min" else -1
        if solution_transform is None or fitness_transform is None:
            self._identity_ = torch.nn.Identity()
        if solution_transform is None:
            solution_transform = self._identity_
        if fitness_transform is None:
            fitness_transform = self._identity_
        callable(
            solution_transform
        ), f"Expect solution transform to be callable, got {solution_transform}"
        callable(fitness_transform), f"Expect fitness transform to be callable, got {fitness_transform}"
        self._solution_transform_ = solution_transform
        self._fitness_transform_ = fitness_transform

    def setup(
        self,
        algorithm: Algorithm,
        problem: Problem,
        monitor: Monitor | None = None,
        device: str | torch.device | int | None = None,
    ):
        """Setup the module with submodule initialization.

        Args:
            algorithm (`Algorithm`): The algorithm to be used in the workflow.
            problem (`Problem`): The problem to be used in the workflow.
            monitors (`Sequence[Monitor] | None`, optional): The monitors to be used in the workflow. Defaults to None. Notice: usually, monitors can only be used when using JIT script mode.
            device (`str | torch.device | int | None`, optional): The device of the workflow. Defaults to None.

        ## Notice:
        The algorithm, problem and monitor will be IN-PLACE transformed to the target device.
        """
        algorithm.to(device=device)
        problem.to(device=device)
        self.algorithm = algorithm
        self._has_init_ = type(algorithm).init_step != Algorithm.init_step
        monitor = (
            Monitor()
            if monitor is None
            else monitor.set_config(opt_direction=self.opt_direction).setup().to(device=device)
        )

        # set algorithm evaluate
        self.algorithm.evaluate = self._evaluate
        self.algorithm._problem_ = problem
        self.algorithm._monitor_ = monitor
        self.algorithm._solution_transform_ = self._solution_transform_
        self.algorithm._fitness_transform_ = self._fitness_transform_
        # for compilation, not used
        self._monitor_ = Monitor()
        self._problem_ = Problem()

    def __getattribute__(self, name: str):
        if name == "_monitor_":
            return self.algorithm._monitor_
        elif name == "_problem_":
            return self.algorithm._problem_
        return super().__getattribute__(name)

    @torch.jit.ignore
    def monitor(self):
        return self.algorithm._monitor_

    @torch.jit.ignore
    def problem(self):
        return self.algorithm._problem_

    def _evaluate(self, population: torch.Tensor) -> torch.Tensor:
        self._monitor_.post_ask(population)
        population = self._solution_transform_(population)
        self._monitor_.pre_eval(population)
        fitness = self._problem_.evaluate(population)
        fitness *= self.opt_direction
        self._monitor_.post_eval(fitness)
        fitness = self._fitness_transform_(fitness)
        self._monitor_.pre_tell(fitness)
        return fitness

    def _step(self, init: bool):
        if init and self._has_init_:
            self.algorithm.init_step()
        else:
            self.algorithm.step()

    def init_step(self):
        """
        Perform the first optimization step of the workflow.

        Calls the `init_step` of the algorithm if overwritten; otherwise, its `step` method will be invoked.
        """
        self._step(init=True)

    def step(self):
        """Perform a single optimization step using the algorithm and the problem."""
        self._step(init=False)
