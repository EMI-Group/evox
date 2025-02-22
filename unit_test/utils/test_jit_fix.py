import unittest

import torch

from evox.utils import switch


class TestJitFix(unittest.TestCase):
    def test_basic_switch(self):
        x = torch.tensor([1, 0, 1], dtype=torch.int)
        y = torch.tensor([[2.0, 2.5], [3.0, 3.5], [4.0, 4.5]]).T.split(1, dim=0)
        y = [a.squeeze(0) for a in y]
        basic_switch = torch.compile(switch)
        z = basic_switch(x, y)
        self.assertIsNotNone(z)

    def test_vmap_switch(self):
        x = torch.randint(low=0, high=10, size=(2, 10), dtype=torch.int)
        y = [torch.rand(2, 10) for _ in range(10)]
        vmap_switch = torch.compile(switch)
        z = vmap_switch(x, y)
        self.assertIsNotNone(z)
