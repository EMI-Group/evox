import unittest

import torch

from evox.problems.so import CEC2022


class TestCEC2022(unittest.TestCase):
    def setUp(self):
        self.dimensionality = [2, 10, 20]
        self.pop_size = 7

    def test_evaluate(self):
        for i in range(1, 13):
            for dimension in self.dimensionality:
                if dimension == 2 and i in [6, 7, 8]:
                    continue
                problem = CEC2022(problem_number=i, dimension=dimension)
                population = torch.randn(self.pop_size, dimension)
                fitness = problem.evaluate(population)
                print(f"The fitness of No.{i} function with {dimension} dimension is {fitness}")

if __name__ == "__main__":
    test = TestCEC2022()
    test.setUp()
    test.test_evaluate()
    # print(CEC2022(12, 20).cf_cal.code)
