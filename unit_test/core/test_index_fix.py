import unittest

import torch

from evox.core import compile, vmap


class TestIndexFix(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")

    def test_get(self):
        def _get_vmap(x: torch.Tensor, index: torch.Tensor, range: torch.Tensor):
            return x[range, index, index]

        mapped = compile(vmap(_get_vmap, in_dims=(0, 0, None)))
        x = torch.rand(3, 2, 5, 5)
        indices = torch.randint(5, (3,))
        print(x)
        print(indices)
        x = mapped(x, indices, torch.arange(2))
        print(x)

    def test_set(self):
        def _set_vmap(x: torch.Tensor, index: torch.Tensor, range: torch.Tensor, value: torch.Tensor):
            x[range, index, index] = value
            return x

        mapped = compile(vmap(_set_vmap, in_dims=(0, 0, None, 0)), fullgraph=True)
        x = torch.rand(3, 2, 5, 5)
        indices = torch.randint(5, (3,))
        values = torch.rand(3, 2)
        print(x)
        print(indices)
        print(values)
        x = mapped(x, indices, torch.arange(2), values)
        print(x)
