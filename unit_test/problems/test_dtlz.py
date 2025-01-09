import unittest

import torch

from evox.core import jit_class
from evox.problems.mo import DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7


class TestDTLZ(unittest.TestCase):
    def setUp(self):
        self.d = 12
        self.m = 3
        self.ref_num = 1000
        self.problems = [DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7]
        self.population = torch.tensor(
            [
                [0.1, 0.5, 0.2, 0.1, 0.5, 0.2, 0.1, 0.5, 0.2, 0.1, 0.5, 0.2],
                [0.8, 0.8, 0.9, 0.8, 0.8, 0.9, 0.8, 0.8, 0.9, 0.8, 0.8, 0.9],
            ]
        )

    def test_evaluate(self):
        for p in self.problems:
            problem = p(d=self.d, m=self.m, ref_num=self.ref_num)
            fitness = problem.evaluate(self.population)
            print("Fitness of the population:")
            print(fitness)
            pf = problem.pf()
            print("Pareto front:")
            print(pf)

    def test_jit_evaluate(self):
        for p in self.problems:
            p = jit_class(p)
            problem = p(d=self.d, m=self.m, ref_num=self.ref_num)
            fitness = problem.evaluate(self.population)
            print("Fitness of the population:")
            print(fitness)
            pf = problem.pf()
            print("Pareto front:")
            print(pf)
