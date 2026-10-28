import torch

from evox.algorithms import DE, ODE, SHADE, CoDE, SaDE
from evox.workflows import StdWorkflow

from .test_base import Sphere, TestBase


class TestDEVariants(TestBase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
        self.pop_size = 10
        self.dim = 4
        self.lb = -10 * torch.ones(self.dim)
        self.ub = 10 * torch.ones(self.dim)

    def test_de_rand(self):
        algo = DE(self.pop_size, self.lb, self.ub, base_vector="rand")
        self.run_all(algo)

    def test_de_best(self):
        algo = DE(self.pop_size, self.lb, self.ub, base_vector="best")
        self.run_all(algo)

    def test_ode_rand(self):
        algo = ODE(self.pop_size, self.lb, self.ub, base_vector="rand")
        self.run_all(algo)

    def test_ode_best(self):
        algo = ODE(self.pop_size, self.lb, self.ub, base_vector="best")
        self.run_all(algo)

    def test_shade(self):
        algo = SHADE(self.pop_size, self.lb, self.ub)
        self.run_all(algo)

    def test_code(self):
        algo = CoDE(self.pop_size, self.lb, self.ub)
        self.run_all(algo)

    def test_sade(self):
        algo = SaDE(self.pop_size, self.lb, self.ub)
        self.run_all(algo)

    def test_shade_adapts_memory(self):
        algo = SHADE(self.pop_size, self.lb, self.ub)
        workflow = StdWorkflow(algo, Sphere())
        memory = algo.Memory_FCR.clone()
        for _ in range(5):
            workflow.step()
        self.assertFalse(torch.equal(algo.Memory_FCR, memory))
        self.assertTrue(torch.isfinite(algo.Memory_FCR).all())

    def test_initial_population_within_bounds(self):
        for algorithm in (SHADE, CoDE, SaDE):
            algo = algorithm(self.pop_size, self.lb, self.ub)
            self.assertTrue(((algo.pop >= self.lb) & (algo.pop <= self.ub)).all(), algorithm.__name__)
