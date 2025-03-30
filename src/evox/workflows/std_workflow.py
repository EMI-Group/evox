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

    def __init__(
        self,
        algorithm: Algorithm,
        problem: Problem,
        monitor: Monitor | None = None,
        opt_direction: str = "min",
        solution_transform: torch.nn.Module | None = None,
        fitness_transform: torch.nn.Module | None = None,
        device: str | torch.device | int | None = None,
        enable_distributed: bool = False,
        group: Any = None,
    ):
        """Initialize the standard workflow with static arguments.

        :param algorithm: The algorithm to be used in the workflow.
        :param problem: The problem to be used in the workflow.
        :param monitors: The monitors to be used in the workflow. Defaults to None.
        :param opt_direction: The optimization direction, can only be "min" or "max". Defaults to "min". If "max", the fitness will be negated prior to `fitness_transform` and monitor.
        :param solution_transform: The solution transformation function. MUST be compile-compatible module/function. Defaults to None.
        :param fitness_transforms: The fitness transformation function. MUST be compile-compatible module/function. Defaults to None.
        :param device: The device of the workflow. Defaults to None.
        :param enable_distributed: Whether to enable distributed workflow. Defaults to False.
        :param group: The group name used in the distributed workflow. Defaults to None.

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
        self.enable_distributed = enable_distributed
        self.group = group

    def get_submodule(self, target: str) -> Any:
        return super().get_submodule(target)

    def _evaluate(self, population: torch.Tensor) -> torch.Tensor:
        self.monitor.post_ask(population)

        if self.enable_distributed:
            rank = torch.distributed.get_rank(group=self.group)
            pop_size = population.size(0)
            world_size = torch.distributed.get_world_size(group=self.group)
            rank = torch.distributed.get_rank(group=self.group)
            population = population.tensor_split(world_size, dim=0)[rank]

        population = self.solution_transform(population)
        self.monitor.pre_eval(population)

        if self.enable_distributed:
            # When using distributed, we need to make sure that the random number generator is forked.
            # Otherwise, since the evaluation process for different individuals is not independent,
            # the random number generator could get polluted.
            with torch.random.fork_rng():
                fitness = self.problem.evaluate(population)

            # contruct a list of tensors to gather all fitness
            all_fitness = torch.zeros(pop_size, *fitness.shape[1:], device=fitness.device, dtype=fitness.dtype)
            all_fitness = list(all_fitness.tensor_split(world_size, dim=0))
            # gather all fitness
            torch.distributed.all_gather(all_fitness, fitness, group=self.group)
            fitness = torch.cat(all_fitness, dim=0)
        else:
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
