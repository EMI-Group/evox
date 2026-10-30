from unittest import TestCase

import torch

from evox.algorithms import CLPSO, CSO, DMSPSOEL, FSPSO, PSO, SLPSOGS, SLPSOUS
from evox.problems.numerical import Ackley
from evox.workflows import EvalMonitor, StdWorkflow

from .test_base import TestBase


class TestPSODeviceTransfer(TestCase):
    def setUp(self):
        self.addCleanup(torch.set_default_device, torch.get_default_device())
        torch.set_default_device("cpu")
        torch.manual_seed(42)

    def test_workflow_device_transfer(self):
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        for device_name in devices:
            with self.subTest(device=device_name):
                device = torch.device(device_name)
                algorithm = PSO(pop_size=100, lb=-32 * torch.ones(10), ub=32 * torch.ones(10))
                problem = Ackley()
                monitor = EvalMonitor()
                workflow = StdWorkflow(algorithm=algorithm, problem=problem, monitor=monitor, device=device)
                for _ in range(100):
                    workflow.step()
                self.assertEqual(algorithm.lb.device, algorithm.pop.device)
                self.assertEqual(algorithm.ub.device, algorithm.pop.device)
                self.assertTrue(torch.isfinite(algorithm.fit).all())

    def test_bound_dtype_transfer(self):
        lb = torch.tensor([-32.0, -8.0])
        ub = torch.tensor([1.0, 32.0])
        algorithm = PSO(pop_size=4, lb=lb, ub=ub)
        algorithm.to(dtype=torch.float64)
        torch.testing.assert_close(algorithm.lb, lb[None, :].double())
        torch.testing.assert_close(algorithm.ub, ub[None, :].double())

    def test_load_legacy_state(self):
        algorithm = PSO(pop_size=4, lb=torch.tensor([-32.0, -8.0]), ub=torch.tensor([1.0, 32.0]))
        legacy_state = {name: value for name, value in algorithm.state_dict().items() if name not in ("lb", "ub")}
        algorithm.load_state_dict(legacy_state)


class TestPSOVariants(TestBase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
        self.pop_size = 10
        self.dim = 4
        self.lb = -10 * torch.ones(self.dim)
        self.ub = 10 * torch.ones(self.dim)

    def test_clpso(self):
        algo = CLPSO(self.pop_size, self.lb, self.ub)
        self.run_all(algo)

    def test_cso(self):
        algo = CSO(self.pop_size, self.lb, self.ub)
        self.run_all(algo)

    def test_dmspsoel(self):
        algo = DMSPSOEL(
            self.lb,
            self.ub,
            self.pop_size // 2,
            9,
            self.pop_size // 2,
            max_iteration=3,
        )
        self.run_algorithm(algo)
        self.run_compiled_algorithm(algo)

    def test_fspso(self):
        algo = FSPSO(self.pop_size, self.lb, self.ub)
        self.run_all(algo)

    def test_pso(self):
        algo = PSO(self.pop_size, self.lb, self.ub)
        self.run_all(algo)

    def test_slpsogs(self):
        algo = SLPSOGS(self.pop_size, self.lb, self.ub)
        self.run_all(algo)

    def test_clpsous(self):
        algo = SLPSOUS(self.pop_size, self.lb, self.ub)
        self.run_all(algo)
