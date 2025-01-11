<<<<<<< HEAD
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
=======
import torch

from evox.problems.numerical import DTLZ1

if __name__ == "__main__":
    # Problem dimensions and objectives
    d = 12
    m = 3
    ref_num = 1000

    # Create an instance of the DTLZ1 problem
    problem = DTLZ1(d=d, m=m, ref_num=ref_num)

    # Generate a random population (100 individuals, each with d features)
    # population = torch.rand(100, d)
    population = torch.tensor(
        [
            [0.1, 0.5, 0.2, 0.1, 0.5, 0.2, 0.1, 0.5, 0.2, 0.1, 0.5, 0.2],
            [0.8, 0.8, 0.9, 0.8, 0.8, 0.9, 0.8, 0.8, 0.9, 0.8, 0.8, 0.9],
        ]
    )

    # Evaluate the population
    fitness = problem.evaluate(population)

    print("Fitness of the population:")
    print(fitness)

    # Get the Pareto front for DTLZ1
    pf = problem.pf()
    print("Pareto front:")
    print(pf)
>>>>>>> d24aa7880432e5f07e6f712b1a852dd3bf7b0c2d
