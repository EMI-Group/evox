import unittest

import torch

from evox.operators.selection import ref_vec_guided


class TestRefVecGuided(unittest.TestCase):
    def setUp(self):
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

    def test_empty_vector_fallback(self):
        """Empty vectors must be filled with the globally best-APD solution.

        Constructed case: solutions clustered in one direction, so most
        reference vectors have no associated solution and would previously be
        NaN-padded (the morobotrol collapse).
        """
        n, m, nv = 10, 2, 6
        x = torch.randn(n, 4)
        # all solutions point near (1, 1) direction -> only associate to
        # vectors close to that direction
        f = torch.stack([torch.rand(n) + 1.0, torch.rand(n) + 1.0], dim=1)
        v = torch.tensor(
            [[0.7071, 0.7071], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [0.7071, 0.7071]]
        )
        theta = torch.tensor(0.2)

        next_x, next_f = ref_vec_guided(x, f, v, theta)
        self.assertEqual(next_x.size(0), nv)
        self.assertTrue(torch.isfinite(next_x).all())
        self.assertTrue(torch.isfinite(next_f).all())
        # every selected solution must come from the input population
        for row in next_f:
            self.assertTrue(torch.any(torch.all(f == row, dim=1)))

        # jit path must agree
        next_x1, next_f1 = self.jit_ref_vec_guided(x, f, v, theta)
        self.assertTrue(torch.isfinite(next_x1).all())
        self.assertTrue(torch.isfinite(next_f1).all())
