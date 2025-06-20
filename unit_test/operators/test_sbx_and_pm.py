from unittest import TestCase

import torch

from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation


class TestOperators(TestCase):
    def setUp(self):
        torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_individuals = 9
        self.n_genes = 10
        self.x = torch.randn(self.n_individuals, self.n_genes)
        self.lb = -torch.ones(self.n_genes)
        self.ub = torch.ones(self.n_genes)

    def test_simulated_binary(self):
        offspring = simulated_binary(self.x)
        self.assertEqual(offspring.size(0), self.x.size(0) // 2 * 2)
        jit_simulated_binary = torch.compile(simulated_binary)
        offspring = jit_simulated_binary(self.x, pro_c=torch.tensor(0.5), dis_c=torch.tensor(20))
        self.assertEqual(offspring.size(0), self.x.size(0) // 2 * 2)

    def test_polynomial_mutation(self):
        offspring = polynomial_mutation(
            self.x,
            lb=self.lb,
            ub=self.ub,
        )
        self.assertEqual(offspring.size(), self.x.size())
        jit_polynomial_mutation = torch.compile(polynomial_mutation)
        offspring = jit_polynomial_mutation(
            self.x,
            lb=self.lb,
            ub=self.ub,
            pro_m=torch.tensor(0.5),
            dis_m=torch.tensor(20),
        )
        self.assertEqual(offspring.size(), self.x.size())
