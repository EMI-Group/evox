import unittest

import torch

from evox.core import compile, vmap
from evox.operators.selection import non_dominate_rank


class TestNonDominate(unittest.TestCase):
    def setUp(self):
        torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
        self.n, self.m = 12, 3
        self.f = torch.randn(self.n, self.m)
        self.rank = non_dominate_rank(self.f)

    def test_compile(self):
        rank = compile(non_dominate_rank)(self.f)
        self.assertTrue(torch.equal(rank, self.rank))

    def test_vmap(self):
        rank = compile(vmap(non_dominate_rank))(torch.stack([self.f] * 5))
        self.assertTrue(torch.equal(rank, torch.stack([self.rank] * 5)))

    def test_vmap_vmap(self):
        rank = compile(vmap(vmap(non_dominate_rank)))(torch.stack([torch.stack([self.f] * 5)] * 3))
        self.assertTrue(torch.equal(rank, torch.stack([torch.stack([self.rank] * 5)] * 3)))
