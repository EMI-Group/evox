from typing import Tuple
import torch

import os
import sys
current_directory = os.getcwd()
if current_directory not in sys.path:
    sys.path.append(current_directory)
    
from src.core import use_state, jit, vmap
from src.utils import TracingWhileLoop


if __name__ == "__main__":
    def loop_body(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return x + 1, y ** 1.05
    
    def loop_cond(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x < 10
    
    while_loop = TracingWhileLoop(loop_cond, loop_body)
    x = torch.tensor(0, dtype=torch.int)
    y = torch.tensor([2.0, 2.5])
    x1, y1 = while_loop.loop(x, y)
    print(x1, y1)
    trace_loop = jit(use_state(lambda: while_loop.loop), trace=True, lazy=False, example_inputs=(x, y))
    x1, y1 = trace_loop(x, y)
    print(x1, y1)
    
    x = torch.tensor([0, 1, 2], dtype=torch.int)
    y = torch.tensor([[2.0, 2.5], [3.0, 3.5], [4.0, 4.5]])
    vmap_loop = jit(vmap(use_state(lambda: while_loop.loop)), trace=True, lazy=False, example_inputs=(x, y))
    x1, y1 = vmap_loop(x, y)
    print(x1, y1)
    
    x = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.int)
    y = torch.tensor([[[2.0, 2.5], [3.0, 3.5], [4.0, 4.5]], [[2.1, 2.2], [3.1, 3.2], [4.1, 4.2]]])
    vmap_loop = jit(vmap(vmap(use_state(lambda: while_loop.loop))), trace=True, lazy=False, example_inputs=(x, y))
    x1, y1 = vmap_loop(x, y)
    print(x1, y1)