import torch

from evox.algorithms import CLPSO, CSO, DMSPSOEL, FSPSO, PSO, SLPSOGS, SLPSOUS

from .test_base import TestBase


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
