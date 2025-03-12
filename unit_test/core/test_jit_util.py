import unittest

import torch

from evox.core import vmap


@vmap
def _single_eval(x: torch.Tensor, index: torch.Tensor, p: float = 2.0, q: torch.Tensor = torch.tensor([0, 1])):
    return (x[index] ** p).sum() * q.sum()


class TestJitUtil(unittest.TestCase):
    def setUp(self):
        self.expected = torch.tensor([4.0] * 10)

    def test_single_eval(self):
        result = _single_eval(2 * torch.ones(10, 2), torch.randint(0, 2, (10,)))
        self.assertTrue(torch.equal(result, self.expected))

    def test_jit_single_eval(self):
        result = torch.compile(_single_eval)(2 * torch.ones(10, 2), torch.randint(0, 2, (10,)))
        self.assertTrue(torch.equal(result, self.expected))
