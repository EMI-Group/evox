from unittest import TestCase

import torch

from evox.operators.crossover import DE_binary_crossover


class TestDEBinaryCrossover(TestCase):
    def setUp(self):
        torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)

    def test_crossover_rate_matches_CR(self):
        # DE_binary_crossover should take the mutated gene with probability CR.
        # A low CR must keep the trial vector close to the current vector.
        pop_size, dim = 2000, 20
        mutation_vector = torch.ones(pop_size, dim)
        current_vector = torch.zeros(pop_size, dim)
        CR = torch.full((pop_size,), 0.1)

        trial_vector = DE_binary_crossover(mutation_vector, current_vector, CR)
        empirical_rate = trial_vector.mean().item()

        self.assertLess(empirical_rate, 0.2)

    def test_high_CR_crosses_over_most_genes(self):
        pop_size, dim = 2000, 20
        mutation_vector = torch.ones(pop_size, dim)
        current_vector = torch.zeros(pop_size, dim)
        CR = torch.full((pop_size,), 0.9)

        trial_vector = DE_binary_crossover(mutation_vector, current_vector, CR)
        empirical_rate = trial_vector.mean().item()

        self.assertGreater(empirical_rate, 0.85)
