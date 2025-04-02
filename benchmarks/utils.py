"""Benchmark the performance of utils functions in EvoX"""

import torch
from torch.profiler import ProfilerActivity, profile

from evox.core import compile, vmap
from evox.utils import switch


def run_switch():
    x = torch.tensor([1, 0, 1], dtype=torch.int)
    y = torch.tensor([[2.0, 2.5], [3.0, 3.5], [4.0, 4.5]]).T.split(1, dim=0)
    y = [a.squeeze(0) for a in y]
    basic_switch = compile(switch)
    z = basic_switch(x, y)
    print(z, flush=True)
    print("\n")

    x = torch.randint(low=0, high=10, size=(1000, 10000), dtype=torch.int, device="cuda")
    y = [torch.rand(1000, 10000, device="cuda") for _ in range(10)]
    vmap_switch = compile(vmap(switch))
    z = vmap_switch(x, y)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for _ in range(1000):
            z = vmap_switch(x, y)
    print(prof.key_averages().table(), flush=True)


if __name__ == "__main__":
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    run_switch()
