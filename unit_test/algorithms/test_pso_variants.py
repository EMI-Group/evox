import torch

from evox.algorithms import CLPSO, CSO, DMSPSOEL, FSPSO, PSO, SLPSOGS, SLPSOUS

from .test_base import TestBase


class TestPSOVariants(TestBase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
        pop_size = 10
        dim = 4
        lb = -10 * torch.ones(dim)
        ub = 10 * torch.ones(dim)
        self.algos = [
            CLPSO(pop_size, lb, ub),
            CSO(pop_size, lb, ub),
            DMSPSOEL(
                lb,
                ub,
                pop_size // 2,
                9,
                pop_size // 2,
                max_iteration=3,
            ),
            FSPSO(pop_size, lb, ub),
            PSO(pop_size, lb, ub),
            SLPSOGS(pop_size, lb, ub),
            SLPSOUS(pop_size, lb, ub),
        ]

    def test_pso_variants(self):
        for algo in self.algos:
            self.run_algorithm(algo)
            self.run_trace_algorithm(algo)

            if not isinstance(algo, DMSPSOEL):
                self.run_vmap_algorithm(algo)
