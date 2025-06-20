import unittest

import torch

from evox.core import Algorithm, Mutable, Parameter, Problem
from evox.metrics import igd
from evox.problems.hpo_wrapper import HPOFitnessMonitor, HPOProblemWrapper
from evox.problems.numerical import DTLZ1
from evox.workflows import StdWorkflow


class BasicProblem(Problem):
    def __init__(self):
        super().__init__()

    def evaluate(self, x: torch.Tensor):
        return (x * x).sum(-1)


class BasicAlgorithm(Algorithm):
    def __init__(self, pop_size: int, lb: torch.Tensor, ub: torch.Tensor, device: torch.device | None = None):
        super().__init__()
        assert lb.ndim == 1 and ub.ndim == 1, (
            f"Lower and upper bounds shall have ndim of 1, got {lb.ndim} and {ub.ndim}"
        )
        assert lb.shape == ub.shape, f"Lower and upper bounds shall have same shape, got {lb.ndim} and {ub.ndim}"
        device = torch.get_default_device() if device is None else device
        self.pop_size = pop_size
        self.dim = lb.shape[0]
        self.hp = Parameter([1.0, 2.0], device=device)
        self.lb = lb.to(device=device)
        self.ub = ub.to(device=device)
        self.pop = Mutable(torch.empty(self.pop_size, lb.shape[0], dtype=lb.dtype, device=lb.device))
        self.fit = Mutable(torch.empty(self.pop_size, dtype=lb.dtype, device=lb.device))

    def step(self):
        pop = torch.rand(self.pop_size, self.dim, dtype=self.lb.dtype, device=self.lb.device)
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
        self.workflow = StdWorkflow(self.algo, self.prob, monitor=self.monitor)
        self.hpo_prob = HPOProblemWrapper(
            iterations=9, num_instances=7, workflow=self.workflow, copy_init_state=True
        )

        self.algo_mo = BasicAlgorithm(10, -10 * torch.ones(2), 10 * torch.ones(2))
        self.prob_mo = DTLZ1(2, 2)
        self.monitor_mo = HPOFitnessMonitor(multi_obj_metric=lambda f: igd(f, self.prob_mo.pf()))
        self.workflow_mo = StdWorkflow(self.algo_mo, self.prob_mo, monitor=self.monitor_mo)
        self.hpo_prob_mo = HPOProblemWrapper(
            iterations=9, num_instances=7, workflow=self.workflow_mo, copy_init_state=True
        )

        self.algo_mo2 = BasicAlgorithm(10, -10 * torch.ones(2), 10 * torch.ones(2))
        self.prob_mo2 = DTLZ1(2, 2)
        self.monitor_mo2 = HPOFitnessMonitor(multi_obj_metric=lambda f: igd(f, self.prob_mo2.pf()))
        self.workflow_mo2 = StdWorkflow(self.algo_mo2, self.prob_mo2, monitor=self.monitor_mo2)
        self.hpo_prob_mo2 = HPOProblemWrapper(
            iterations=9, num_instances=7, workflow=self.workflow_mo2, copy_init_state=True
        )

    def test_get_init_params(self):
        params = self.hpo_prob.get_init_params()
        self.assertIn("algorithm.hp", params)

    def test_evaluate(self):
        params = self.hpo_prob.get_init_params()
        params["algorithm.hp"] = Parameter(torch.rand(7, 2), requires_grad=False)
        result = self.hpo_prob.evaluate(params)
        self.assertIsInstance(result, torch.Tensor)

    def test_outer_workflow(self):
        class solution_transform(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                return {"algorithm.hp": x}

        outer_algo = BasicAlgorithm(7, -10 * torch.ones(2), 10 * torch.ones(2))
        outer_workflow = StdWorkflow(outer_algo, self.hpo_prob, solution_transform=solution_transform())
        outer_workflow.init_step()
        self.assertIsInstance(outer_workflow.algorithm.fit, torch.Tensor)

    def test_outer_workflow_mo(self):
        class solution_transform(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                return {"algorithm.hp": x}

        outer_algo = BasicAlgorithm(7, -10 * torch.ones(2), 10 * torch.ones(2))
        outer_workflow = StdWorkflow(outer_algo, self.hpo_prob_mo, solution_transform=solution_transform())
        outer_workflow.init_step()
        self.assertIsInstance(outer_workflow.algorithm.fit, torch.Tensor)

    def test_outer_workflow_mo_repeat(self):
        class solution_transform(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                return {"algorithm.hp": x}

        outer_algo = BasicAlgorithm(7, -10 * torch.ones(2), 10 * torch.ones(2))
        outer_workflow = StdWorkflow(outer_algo, self.hpo_prob_mo2, solution_transform=solution_transform())
        outer_workflow.init_step()
        self.assertIsInstance(outer_workflow.algorithm.fit, torch.Tensor)
