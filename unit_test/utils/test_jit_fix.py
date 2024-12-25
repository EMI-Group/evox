import os
import sys
current_directory = os.getcwd()
if current_directory not in sys.path:
    sys.path.append(current_directory)
    
import torch
from torch.profiler import profile, ProfilerActivity

from src.core import jit, vmap
from src.utils import switch


if __name__ == "__main__":
    x = torch.tensor([1, 0, 1], dtype=torch.int)
    y = torch.tensor([[2.0, 2.5], [3.0, 3.5], [4.0, 4.5]]).T.split(1, dim=0)
    y = [a.squeeze(0) for a in y]
    print(y)
    basic_switch = jit(switch, trace=False)
    z = basic_switch(x, y)
    print(z)
    
    x = torch.randint(low=0, high=10, size=(1000, 10000), dtype=torch.int, device="cuda")
    y = [torch.rand(1000, 10000, device="cuda") for _ in range(10)]
    vmap_switch = jit(switch, trace=False)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True
    ) as prof:
        for _ in range(1000):
            z = vmap_switch(x, y)
    print(prof.key_averages().table())
    
    x = torch.tensor([[1, 0, 1], [1, 1, 0]], dtype=torch.int)
    y = torch.tensor([[[2.0, 2.5], [3.0, 3.5], [4.0, 4.5]], [[2.0, 2.5], [3.0, 3.5], [4.0, 4.5]]]).transpose(1, 2).split(1, dim=1)
    y = [a.squeeze() for a in y]
    print(y)
    vmap_switch = jit(vmap(switch, trace=False), trace=True, lazy=False, example_inputs=(x, y))
    z = vmap_switch(x, y)
    print(z)
    
    x = torch.randint(low=0, high=10, size=(1000, 10000), dtype=torch.int, device="cuda")
    y = [torch.rand(1000, 10000, device="cuda") for _ in range(10)]
    vmap_switch = jit(vmap(switch, trace=False), trace=True, lazy=False, example_inputs=(x, y))
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True
    ) as prof:
        for _ in range(1000):
            z = vmap_switch(x, y)
    print(prof.key_averages().table())