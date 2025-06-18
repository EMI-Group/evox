import unittest

import torch
from CEC2022_by_P_N_Suganthan import cec2022_func

from evox.problems.numerical import CEC2022


class TestCEC2022(unittest.TestCase):
    def setUp(self):
        self.dimensionality = [2, 10, 20]
        self.pop_size = 100
        torch.manual_seed(42)

    def test_evaluate(self):
        for i in range(1, 13):
            for dimension in self.dimensionality:
                if dimension == 2 and i in [6, 7, 8]:
                    continue

                population = torch.rand(self.pop_size, dimension)
                problem = CEC2022(problem_number=i, dimension=dimension)
                fitness = problem.evaluate(population)
                target = cec2022_func(func_num=i)
                target_fitness = target.values(population.detach().cpu().numpy())
                self.assertTrue(torch.allclose(fitness, torch.as_tensor(target_fitness, dtype=fitness.dtype)))
