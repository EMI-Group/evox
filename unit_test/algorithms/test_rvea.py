import torch

from evox.algorithms import RVEA

from .test_base import TestBase


class TestMOAlgorithms(TestBase):
    def setUp(self):
        pop_size = 10
        lb = torch.zeros(12)
        ub = torch.ones(12)
        self.algos = [
            RVEA(pop_size, n_objs=3, lb=lb, ub=ub),
        ]

    def test_de_variants(self):
        for algo in self.algos:
            self.run_mo_algorithm(algo)
            self.run_vmap_mo_algorithm(algo)
