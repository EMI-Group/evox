import unittest
import torch
from torch import nn
from evox.core import jit_class, Problem, Algorithm, trace_impl, Parameter
from evox.workflows import StdWorkflow
from evox.problems.hpo_wrapper import HPOProblemWrapper, HPOFitnessMonitor


@jit_class
class BasicProblem(Problem):
    def __init__(self):
        super().__init__()

    def evaluate(self, x: torch.Tensor):
        return (x * x).sum(-1)


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
        self.hp = Parameter([1.0, 2.0])
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
        pop = pop * self.hp[0]
        self.pop.copy_(pop)
        self.fit.copy_(self.evaluate(pop))

    @trace_impl(step)
    def trace_step(self):
        pop = torch.rand(
            self.pop_size, self.dim, dtype=self.lb.dtype, device=self.lb.device
        )
        pop = pop * (self.ub - self.lb)[None, :] + self.lb[None, :]
        pop = pop * self.hp[0]
        self.pop = pop
        self.fit = self.evaluate(pop)


class TestHPOWrapper(unittest.TestCase):
    def setUp(self):
        torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
        self.algo = BasicAlgorithm(10, -10 * torch.ones(2), 10 * torch.ones(2))
        self.prob = BasicProblem()
        self.monitor = HPOFitnessMonitor()
        self.monitor.setup()
        self.workflow = StdWorkflow()
        self.workflow.setup(self.algo, self.prob, monitor=self.monitor)
        self.hpo_prob = HPOProblemWrapper(
            iterations=9, num_instances=7, workflow=self.workflow, copy_init_state=True
        )

    def test_extract_parameters(self):
        params = HPOProblemWrapper.extract_parameters(self.hpo_prob.init_state)
        self.assertIn("self.algorithm.hp", params)

    def test_evaluate(self):
        params = HPOProblemWrapper.extract_parameters(self.hpo_prob.init_state)
        params["self.algorithm.hp"] = torch.nn.Parameter(
            torch.rand(7, 2), requires_grad=False
        )
        result = self.hpo_prob.evaluate(params)
        self.assertIsInstance(result, torch.Tensor)

    def test_outer_workflow(self):
        class solution_transform(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                return {"self.algorithm.hp": x}

        outer_algo = BasicAlgorithm(7, -10 * torch.ones(2), 10 * torch.ones(2))
        outer_workflow = StdWorkflow()
        outer_workflow.setup(
            outer_algo, self.hpo_prob, solution_transform=solution_transform()
        )
        outer_workflow.init_step()
        self.assertIsInstance(outer_workflow.algorithm.fit, torch.Tensor)
