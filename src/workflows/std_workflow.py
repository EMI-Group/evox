import sys
from typing import Sequence, Callable, Any

sys.path.append(__file__ + "/../..")

import torch
from core import ModuleBase, Algorithm, Problem, Workflow, Monitor, jit_class


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
            opt_direction (`str`, optional): The optimization direction, can only be "min" or "max". Defaults to "min".
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


# Test
if __name__ == "__main__":
    from torch import nn
    from core import vmap, trace_impl, batched_random, use_state, jit
    from eval_monitor import EvalMonitor

    @jit_class
    class BasicProblem(Problem):

        def __init__(self):
            super().__init__()
            self._eval_fn = vmap(BasicProblem._single_eval, trace=False)
            self._eval_fn_traced = vmap(BasicProblem._single_eval, example_ndim=2)

        def _single_eval(x: torch.Tensor, p: float = 2.0):
            return (x**p).sum()

        def evaluate(self, pop: torch.Tensor):
            return self._eval_fn_traced(pop)

        @trace_impl(evaluate)
        def trace_evaluate(self, pop: torch.Tensor):
            return self._eval_fn(pop)

    @jit_class
    class BasicAlgorithm(Algorithm):

        def __init__(self, pop_size: int):
            super().__init__()
            self.pop_size = pop_size

        def setup(self, lb: torch.Tensor, ub: torch.Tensor):
            assert (
                lb.ndim == 1 and ub.ndim == 1
            ), f"Lower and upper bounds shall have ndim of 1, got {lb.ndim} and {ub.ndim}"
            assert (
                lb.shape == ub.shape
            ), f"Lower and upper bounds shall have same shape, got {lb.ndim} and {ub.ndim}"
            self.lb = lb
            self.ub = ub
            self.dim = lb.shape[0]
            self.pop = nn.Buffer(
                torch.empty(self.pop_size, lb.shape[0], dtype=lb.dtype, device=lb.device)
            )
            self.fit = nn.Buffer(torch.empty(self.pop_size, dtype=lb.dtype, device=lb.device))
            return self

        def step(self):
            pop = torch.rand(self.pop_size, self.dim, dtype=self.lb.dtype, device=self.lb.device)
            pop = pop * (self.ub - self.lb)[None, :] + self.lb[None, :]
            self.pop.copy_(pop)
            self.fit.copy_(self.evaluate(pop))

        @trace_impl(step)
        def trace_step(self):
            pop = batched_random(
                torch.rand, self.pop_size, self.dim, dtype=self.lb.dtype, device=self.lb.device
            )
            pop = pop * (self.ub - self.lb)[None, :] + self.lb[None, :]
            self.pop = pop
            self.fit = self.evaluate(pop)

    # basic
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    algo = BasicAlgorithm(10)
    algo.setup(-10 * torch.ones(2), 10 * torch.ones(2))
    prob = BasicProblem()
    workflow = StdWorkflow()
    workflow.setup(algo, prob)

    # # classic workflow
    # class solution_transform(nn.Module):
    #     def forward(self, x: torch.Tensor):
    #         return x / 5
    # class fitness_transform(nn.Module):
    #     def forward(self, f: torch.Tensor):
    #         return -f
    # monitor = EvalMonitor(full_sol_history=True)
    # workflow = StdWorkflow(solution_transform=solution_transform(), fitness_transform=fitness_transform())
    # workflow.setup(algo, prob, monitor=monitor)
    # print(workflow.step.inlined_graph)
    # workflow.step()
    # monitor = workflow.monitor()
    # print(monitor.topk_fitness)
    # workflow.step()
    # print(monitor.topk_fitness)
    # workflow.step()
    # print(monitor.topk_fitness)

    # # stateful workflow
    # state_step = use_state(lambda: workflow.step, True)
    # print(state_step.init_state())
    # jit_step = jit(state_step, trace=True, example_inputs=(state_step.init_state(),))
    # jit_step(state_step.init_state())
    # print(jit_step(state_step.init_state()))

    # # vmap workflow
    # state_step = use_state(lambda: workflow.step, True)
    # vmap_state_step = vmap(state_step)
    # state = vmap_state_step.init_state(3)
    # print(state)
    # jit_state_step = jit(vmap_state_step, trace=True, lazy=True)
    # print(jit_state_step(state))
