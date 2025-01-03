import unittest
import torch
import torch.nn as nn
from evox.core import (
    vmap,
    trace_impl,
    use_state,
    jit,
    jit_class,
    Algorithm,
    Problem,
)
from evox.workflows import StdWorkflow, EvalMonitor


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
        self.fit = nn.Buffer(
            torch.empty(self.pop_size, dtype=lb.dtype, device=lb.device)
        )

    def step(self):
        pop = torch.rand(
            self.pop_size, self.dim, dtype=self.lb.dtype, device=self.lb.device
        )
        pop = pop * (self.ub - self.lb)[None, :] + self.lb[None, :]
        self.pop.copy_(pop)
        self.fit.copy_(self.evaluate(pop))

    @trace_impl(step)
    def trace_step(self):
        pop = torch.rand(
            self.pop_size, self.dim, dtype=self.lb.dtype, device=self.lb.device
        )
        pop = pop * (self.ub - self.lb)[None, :] + self.lb[None, :]
        self.pop = pop
        self.fit = self.evaluate(pop)


class TestStdWorkflow(unittest.TestCase):

    def setUp(self):
        torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
        self.algo = BasicAlgorithm(10, -10 * torch.ones(2), 10 * torch.ones(2))
        self.prob = BasicProblem()
        self.workflow = StdWorkflow()
        self.workflow.setup(self.algo, self.prob)

    def test_basic_workflow(self):
        state_step = use_state(lambda: self.workflow.step)
        self.assertIsNotNone(state_step.init_state())
        jit_step = jit(
            state_step, trace=True, example_inputs=(state_step.init_state(),)
        )
        self.assertIsNotNone(jit_step(state_step.init_state()))

    def test_vmap_workflow(self):
        state_step = use_state(lambda: self.workflow.step)
        vmap_state_step = vmap(state_step)
        state = vmap_state_step.init_state(3)
        self.assertIsNotNone(state)
        jit_state_step = jit(vmap_state_step, trace=True, lazy=True)
        self.assertIsNotNone(jit_state_step(state))

    def test_classic_workflow(self):
        class solution_transform(nn.Module):
            def forward(self, x: torch.Tensor):
                return x / 5

        class fitness_transform(nn.Module):
            def forward(self, f: torch.Tensor):
                return -f

        monitor = EvalMonitor(full_sol_history=True)
        monitor = monitor.setup()
        workflow = StdWorkflow()
        workflow.setup(
            self.algo,
            self.prob,
            solution_transform=solution_transform(),
            fitness_transform=fitness_transform(),
            monitor=monitor,
        )
        workflow.init_step()
        monitor = workflow.get_submodule("monitor")
        self.assertIsNotNone(monitor.topk_fitness)
        workflow.step()
        self.assertIsNotNone(monitor.topk_fitness)
        workflow.step()
        self.assertIsNotNone(monitor.topk_fitness)
