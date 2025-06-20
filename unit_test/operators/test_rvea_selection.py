import unittest

import torch

from evox.operators.selection import ref_vec_guided


class TestRefVecGuided(unittest.TestCase):
    def setUp(self):
        torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
        self.n, self.m, self.nv = 12, 4, 5
        self.x = torch.randn(self.n, 10)
        self.f = torch.randn(self.n, self.m)
        self.f[1] = torch.tensor([float("nan")] * self.m)

        self.v = torch.randn(self.nv, self.m)
        self.theta = torch.tensor(0.5)

        self.jit_ref_vec_guided = torch.compile(ref_vec_guided)

    def test_ref_vec_guided(self):
        next_x, next_f = ref_vec_guided(self.x, self.f, self.v, self.theta)
        self.assertEqual(next_x.size(0), self.nv)
        self.assertEqual(next_f.size(0), self.nv)
        next_x1, next_f1 = self.jit_ref_vec_guided(self.x, self.f, self.v, self.theta)
        self.assertEqual(next_x1.size(0), self.nv)
        self.assertEqual(next_f1.size(0), self.nv)
