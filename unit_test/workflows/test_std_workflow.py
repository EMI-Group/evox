import unittest

import torch
import torch.nn as nn

from evox.core import Algorithm, Mutable, Problem, use_state, vmap
from evox.workflows import EvalMonitor, StdWorkflow


class BasicProblem(Problem):
    def __init__(self):
        super().__init__()
        self._eval_fn = vmap(BasicProblem._single_eval)

    def _single_eval(x: torch.Tensor, p: float = 2.0):
        return (x**p).sum()

    def evaluate(self, pop: torch.Tensor):
        return self._eval_fn(pop)


class BasicAlgorithm(Algorithm):
    def __init__(self, pop_size: int, lb: torch.Tensor, ub: torch.Tensor):
        super().__init__()
        assert lb.ndim == 1 and ub.ndim == 1, f"Lower and upper bounds shall have ndim of 1, got {lb.ndim} and {ub.ndim}"
        assert lb.shape == ub.shape, f"Lower and upper bounds shall have same shape, got {lb.ndim} and {ub.ndim}"
        self.pop_size = pop_size
        self.lb = lb
        self.ub = ub
        self.dim = lb.shape[0]
        self.pop = Mutable(torch.empty(self.pop_size, lb.shape[0], dtype=lb.dtype, device=lb.device))
        self.fit = Mutable(torch.empty(self.pop_size, dtype=lb.dtype, device=lb.device))

    def step(self):
        pop = torch.rand(self.pop_size, self.dim, dtype=self.lb.dtype, device=self.lb.device)
        pop = pop * (self.ub - self.lb)[None, :] + self.lb[None, :]
        self.pop.copy_(pop)
        self.fit.copy_(self.evaluate(pop))


class TestStdWorkflow(unittest.TestCase):
    def setUp(self):
        torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
        self.algo = BasicAlgorithm(10, -10 * torch.ones(2), 10 * torch.ones(2))
        self.prob = BasicProblem()
        self.workflow = StdWorkflow()
        self.workflow.setup(self.algo, self.prob)

    def test_basic_workflow(self):
        compiled_step = torch.compile(self.workflow.step)
        for _ in range(3):
            compiled_step()

    def test_vmap_workflow(self):
        state_step = use_state(self.workflow.step)
        vmap_state_step = vmap(state_step, randomness="different")
        state = torch.func.stack_module_state([self.workflow] * 3)
        self.assertIsNotNone(state)
        compiled_state_step = torch.compile(vmap_state_step)
        self.assertIsNotNone(compiled_state_step(state))

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
        # test the plot function
        self.assertIsNotNone(monitor.plot())
