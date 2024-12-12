import sys
from typing import Sequence, Tuple, Callable, Any

sys.path.append(__file__ + "/../..")

import torch
from core import ModuleBase, Algorithm, Problem, Workflow, Monitor, jit_class


@jit_class
class StdWorkflow(Workflow):
    """The standard workflow"""

    def __init__(
        self,
        opt_direction: str = "min",
        solution_transforms: Sequence[Callable[[torch.Tensor], torch.Tensor | Any]] | None = None,
        fitness_transforms: Sequence[Callable[[torch.Tensor], torch.Tensor]] | None = None,
    ):
        """Initialize the standard workflow with static arguments.

        Args:
            opt_direction (`str`, optional): The optimization direction, can only be "min" or "max". Defaults to "min".
            solution_transforms (`Sequence[Callable[[torch.Tensor], torch.Tensor | Any]] | None`, optional): The sequence of solution transformation functions. MUST be JIT-compatible function(s) for JIT trace mode or JITed ones for JIT script mode (default mode). Defaults to None.
            fitness_transforms (`Sequence[Callable[[torch.Tensor], torch.Tensor]] | None`, optional): The sequence of fitness transformation functions. MUST be JIT-compatible function(s) for JIT trace mode or JITed ones for JIT script mode (default mode). Defaults to None.
        """
        super().__init__()
        assert opt_direction in [
            "min",
            "max",
        ], f"Expect optimization direction to be `min` or `max`, got {opt_direction}"
        self.opt_direction = 1 if opt_direction == "min" else -1
        if solution_transforms is None:
            solution_transforms = ()
        if fitness_transforms is None:
            fitness_transforms = ()
        for trans in solution_transforms:
            assert callable(trans), f"Expect solution transforms to be callable, got {trans}"
        for trans in fitness_transforms:
            assert callable(trans), f"Expect fitness transforms to be callable, got {trans}"
        self.solution_transforms = tuple(solution_transforms)
        self.fitness_transforms = tuple(fitness_transforms)

    def setup(
        self,
        algorithm: Algorithm,
        problem: Problem,
        monitors: Sequence[Monitor] | None = None,
        device: str | torch.device | int | None = None,
    ):
        """Setup the module with submodule initialization.

        Args:
            algorithm (`Algorithm`): The algorithm to be used in the workflow.
            problem (`Problem`): The problem to be used in the workflow.
            monitors (`Sequence[Monitor] | None`, optional): The monitors to be used in the workflow. Defaults to None.
            device (`str | torch.device | int | None`, optional): The device of the workflow. Defaults to None.

        ## Notice:
        The algorithm, problem and monitors will be IN-PLACE transformed to the target device.
        """
        algorithm.to(device=device)
        problem.to(device=device)
        self.algorithm = algorithm
        self.problem = problem
        monitors = (
            ()
            if monitors is None
            else tuple(
                m.set_config({"opt_direction": self.opt_direction, "multi_obj": problem.num_obj > 1})
                .setup()
                .to(device=device)
                for m in monitors
            )
        )
        self.monitors: Tuple[Monitor] | ModuleBase
        self.add_mutable("monitors", monitors)
        self._has_init = (
            type(algorithm).init_ask != Algorithm.init_ask
            or type(algorithm).init_tell != Algorithm.init_tell
        )

    def init_step(self):
        """The initial step of the workflow. Does nothing if the algorithm does not contain a specific `init_ask` and `init_tell`."""
        if not self._has_init:
            return
        # ask
        for monitor in self.monitors:
            monitor.pre_ask()
        population = self.algorithm.init_ask()  ####
        for monitor in self.monitors:
            monitor.post_ask(population)
        # transform population
        for trans in self.solution_transforms:
            population = trans(population)
        # evaluate
        for monitor in self.monitors:
            monitor.pre_eval(population)
        fitness = self.problem.evaluate(population)  ####
        for monitor in self.monitors:
            monitor.post_eval(fitness)
        # transform fitness
        fitness *= self.opt_direction
        for trans in self.fitness_transforms:
            fitness = trans(fitness)
        # tell
        for monitor in self.monitors:
            monitor.pre_tell(fitness)
        self.algorithm.init_tell(fitness)  ####
        for monitor in self.monitors:
            monitor.post_tell()
        # final
        for monitor in self.monitors:
            monitor.post_step()

    def step(self):
        """The general step of the workflow."""
        # ask
        for monitor in self.monitors:
            monitor.pre_ask()
        population = self.algorithm.ask()  ####
        for monitor in self.monitors:
            monitor.post_ask(population)
        # transform population
        for trans in self.solution_transforms:
            population = trans(population)
        # evaluate
        for monitor in self.monitors:
            monitor.pre_eval(population)
        fitness = self.problem.evaluate(population)  ####
        for monitor in self.monitors:
            monitor.post_eval(fitness)
        # transform fitness
        fitness *= self.opt_direction
        for trans in self.fitness_transforms:
            fitness = trans(fitness)
        # tell
        for monitor in self.monitors:
            monitor.pre_tell(fitness)
        self.algorithm.tell(fitness)  ####
        for monitor in self.monitors:
            monitor.post_tell()
        # final
        for monitor in self.monitors:
            monitor.post_step()


# Test
if __name__ == "__main__":
    from torch import nn
    from core import vmap, trace_impl, batched_random, use_state, jit

    @jit_class
    class BasicProblem(Problem):

        def __init__(self):
            super().__init__(num_objective=1)
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
            super().__init__(pop_size)

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

        def ask(self):
            pop = torch.rand(self.pop_size, self.dim, dtype=self.lb.dtype, device=self.lb.device)
            pop = pop * (self.ub - self.lb)[None, :] + self.lb[None, :]
            self.pop.copy_(pop)
            return self.pop

        def tell(self, fitness: torch.Tensor):
            self.fit.copy_(fitness)

        @trace_impl(ask)
        def trace_ask(self):
            pop = batched_random(
                torch.rand, self.pop_size, self.dim, dtype=self.lb.dtype, device=self.lb.device
            )
            pop = pop * (self.ub - self.lb)[None, :] + self.lb[None, :]
            self.pop = pop
            return self.pop

        @trace_impl(tell)
        def trace_tell(self, fitness: torch.Tensor):
            self.fit = fitness

    # basic
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    algo = BasicAlgorithm(10)
    algo.setup(-10 * torch.ones(2), 10 * torch.ones(2))
    prob = BasicProblem()
    workflow = StdWorkflow()
    workflow.setup(
        algo,
        prob,
    )

    # classic workflow
    print(workflow.step.inlined_graph)
    workflow.init_step()
    print(workflow.algorithm.fit)
    workflow.step()
    print(workflow.algorithm.fit)
    workflow = StdWorkflow()
    workflow.setup(algo, prob)
    print(workflow.algorithm.fit)

    # stateful workflow
    state_step = use_state(lambda: workflow.step, True)
    print(state_step.init_state())
    jit_step = jit(state_step, trace=True, example_inputs=(state_step.init_state(),))
    jit_step(state_step.init_state())
    print(jit_step(state_step.init_state()))

    # vmap workflow
    init_state_step = use_state(lambda: workflow.init_step, True)
    vmap_init_state_step = vmap(init_state_step)
    jit_state_step = jit(vmap_init_state_step, trace=True, lazy=True)
    step1_state = jit_state_step(vmap_init_state_step.init_state(3))
    print(step1_state)
    state_step = use_state(lambda: workflow.step, True)
    vmap_state_step = vmap(state_step, batched_state=step1_state)
    jit_state_step = jit(vmap_state_step, trace=True, lazy=True)
    print(jit_state_step(step1_state))
