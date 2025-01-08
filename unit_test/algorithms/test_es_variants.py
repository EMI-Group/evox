import torch

from evox.algorithms import OpenES

from .test_base import TestBase


class TestESVariants(TestBase):
    def setUp(self):
        pop_size = 10
        dim = 4
        lb = -10 * torch.ones(dim)
        ub = 10 * torch.ones(dim)
        self.algos = [
            OpenES(
                pop_size=pop_size,
                center_init=torch.rand(dim) * (ub - lb) + lb,
                learning_rate=1e-2,
                noise_stdev=5,
                optimizer=None,
            ),
            OpenES(
                pop_size=pop_size,
                center_init=torch.rand(dim) * (ub - lb) + lb,
                learning_rate=1e-2,
                noise_stdev=5,
                optimizer="adam",
            ),
        ]

    def test_es_variants(self):
        for algo in self.algos:
            self.run_algorithm(algo)
            self.run_vmap_algorithm(algo)
