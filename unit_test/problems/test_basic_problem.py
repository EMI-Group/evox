import unittest

import torch

from evox.problems.so import Ackley, BentCigar, Discus, Ellips, Griewank, Happycat, Hgbat, Katsuura, Levy, Rastrigin, Rosenbrock, SchafferF7, Schwefel, Sphere, Zakharov


class TestBasic(unittest.TestCase):
    pass
    def setUp(self, dimensions: list =[1000], pop_size: int = 7):
        self.dimensions = dimensions
        self.pop_size = pop_size
        self.problems = [
            Ackley(),
            BentCigar(),
            Discus(),
            Ellips(),
            Griewank(),
            Happycat(),
            Hgbat(),
            Katsuura(),
            Levy(),
            Rastrigin(),
            Rosenbrock(),
            SchafferF7(),
            Schwefel(),
            Sphere(),
            Zakharov(),
        ]

    def test_evaluate(self):
        for problem in self.problems:
            for dimension in self.dimensions:
                population = torch.randn(self.pop_size, dimension)
                fitness = problem.evaluate(population)
                print(f"The fitness of {problem.original_name} function with {dimension} dimension is {fitness}")

if __name__ == "__main__":
    test = TestBasic()
    test.setUp()
    test.test_evaluate()
