import unittest

import torch

from evox.problems.numerical import Ackley, Ellipsoid, Griewank, Rastrigin, Rosenbrock, Schwefel


class TestBasic(unittest.TestCase):
    def setUp(self, dimensions: list = [10], pop_size: int = 7):
        torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
        self.dimensions = dimensions
        self.pop_size = pop_size
        self.problems = [
            Ackley,
            Griewank,
            Rastrigin,
            Rosenbrock,
            Schwefel,
            Ellipsoid,
        ]

    def test_evaluate(self):
        for problem in self.problems:
            for dimension in self.dimensions:
                problem = problem(shift=torch.rand(dimension), affine=torch.rand(dimension, dimension))
                population = torch.randn(self.pop_size, dimension)
                fitness = problem.evaluate(population)
                print(
                    f"The fitness of {problem.__class__.__name__} function with {dimension} dimension is {fitness}"
                )
