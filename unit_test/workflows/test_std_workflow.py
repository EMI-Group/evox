import torch
import torch.nn as nn

import os
import sys

current_directory = os.getcwd()
if current_directory not in sys.path:
    sys.path.append(current_directory)

from src.core import (
    vmap,
    trace_impl,
    batched_random,
    use_state,
    jit,
    jit_class,
    Algorithm,
    Problem,
)
from src.workflows import StdWorkflow, EvalMonitor


if __name__ == "__main__":

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

        def __init__(self, pop_size: int, lb: torch.Tensor, ub: torch.Tensor):
            super().__init__()
            assert (
                lb.ndim == 1 and ub.ndim == 1
            ), f"Lower and upper bounds shall have ndim of 1, got {lb.ndim} and {ub.ndim}"
            assert (
                lb.shape == ub.shape
            ), f"Lower and upper bounds shall have same shape, got {lb.ndim} and {ub.ndim}"
            self.pop_size = pop_size
            self.lb = lb
            self.ub = ub
            self.dim = lb.shape[0]
            self.pop = nn.Buffer(
                torch.empty(self.pop_size, lb.shape[0], dtype=lb.dtype, device=lb.device)
            )
            self.fit = nn.Buffer(torch.empty(self.pop_size, dtype=lb.dtype, device=lb.device))

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
    algo = BasicAlgorithm(10, -10 * torch.ones(2), 10 * torch.ones(2))
    prob = BasicProblem()
    workflow = StdWorkflow()
    workflow.setup(algo, prob)

    # stateful workflow
    state_step = use_state(lambda: workflow.step)
    print(state_step.init_state())
    jit_step = jit(state_step, trace=True, example_inputs=(state_step.init_state(),))
    jit_step(state_step.init_state())
    print(jit_step(state_step.init_state()))

    # vmap workflow
    state_step = use_state(lambda: workflow.step)
    vmap_state_step = vmap(state_step)
    state = vmap_state_step.init_state(3)
    print(state)
    jit_state_step = jit(vmap_state_step, trace=True, lazy=True)
    print(jit_state_step(state))

    # classic workflow
    class solution_transform(nn.Module):
        def forward(self, x: torch.Tensor):
            return x / 5

    class fitness_transform(nn.Module):
        def forward(self, f: torch.Tensor):
            return -f

    monitor = EvalMonitor(full_sol_history=True)
    workflow = StdWorkflow()
    workflow.setup(
        algo,
        prob,
        solution_transform=solution_transform(),
        fitness_transform=fitness_transform(),
        monitor=monitor,
    )
    workflow.init_step()
    monitor = workflow.get_submodule("monitor")
    print(monitor.topk_fitness)
    workflow.step()
    print(monitor.topk_fitness)
    workflow.step()
    print(monitor.topk_fitness)
