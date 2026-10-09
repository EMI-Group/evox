import unittest

import torch

from evox.metrics import hv


class TestHV(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.ref = torch.tensor([1.0, 1.0])

    def test_dominating_points(self):
        # exact hypervolume of these two points is 0.47
        objs = torch.tensor([[0.2, 0.6], [0.5, 0.3]])
        self.assertAlmostEqual(hv(objs, self.ref, 400000).item(), 0.47, delta=0.01)

    def test_point_not_dominating_ref(self):
        # [1.5, 0.5] is worse than the reference point on the first objective,
        # so it does not add to the hypervolume
        objs = torch.tensor([[0.2, 0.6], [1.5, 0.5]])
        self.assertAlmostEqual(hv(objs, self.ref, 400000).item(), 0.32, delta=0.01)
        self.assertEqual(hv(torch.tensor([[1.5, 0.5]]), self.ref).item(), 0.0)
