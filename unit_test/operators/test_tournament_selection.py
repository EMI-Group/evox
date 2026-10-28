import unittest

import torch

from evox.operators.selection import tournament_selection, tournament_selection_multifit


class TestTournamentSelection(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
        self.fitness = torch.rand(6)

    def test_returns_one_index_per_round(self):
        for n_round in (1, 3):
            selected = tournament_selection(n_round, self.fitness)
            self.assertEqual(selected.shape, (n_round,))
            selected = tournament_selection_multifit(n_round, [self.fitness, -self.fitness])
            self.assertEqual(selected.shape, (n_round,))

    def test_picks_the_tournament_winner(self):
        selected = tournament_selection(100, self.fitness, tournament_size=len(self.fitness) * 10)
        self.assertTrue((selected == torch.argmin(self.fitness)).all())
