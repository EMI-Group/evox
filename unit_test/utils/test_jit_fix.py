import unittest
import torch
from torch.profiler import profile, ProfilerActivity
from evox.core import jit, vmap
from evox.utils import switch


class TestJitFix(unittest.TestCase):
    def test_basic_switch(self):
        x = torch.tensor([1, 0, 1], dtype=torch.int)
        y = torch.tensor([[2.0, 2.5], [3.0, 3.5], [4.0, 4.5]]).T.split(1, dim=0)
        y = [a.squeeze(0) for a in y]
        basic_switch = jit(switch, trace=False, lazy=True)
        z = basic_switch(x, y)
        self.assertIsNotNone(z)
        print(z)

    def test_vmap_switch(self):
        x = torch.randint(
            low=0, high=10, size=(1000, 10000), dtype=torch.int, device="cuda"
        )
        y = [torch.rand(1000, 10000, device="cuda") for _ in range(10)]
        vmap_switch = jit(switch, trace=False)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            for _ in range(1000):
                z = vmap_switch(x, y)
        self.assertIsNotNone(z)
        print(prof.key_averages().table())
