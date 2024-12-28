from typing import List, Tuple
import torch

import os
import sys
current_directory = os.getcwd()
if current_directory not in sys.path:
    sys.path.append(current_directory)
    
from src.core import use_state, jit, vmap
from src.utils import TracingWhile, TracingCond


if __name__ == "__main__":
    def loop_body(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return x + 1, y ** 1.05
    
    def loop_cond(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x < 10
    
    while_loop = TracingWhile(loop_cond, loop_body)
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
    
    print("-" * 100)
    
    def true_fn(x: torch.Tensor, y: torch.Tensor) -> List[torch.Tensor]:
        return [x + 1, y ** 1.05]

    def false_fn(x: torch.Tensor, y: torch.Tensor) -> List[torch.Tensor]:
        return [x - 1, y ** 0.95]
    
    if_else = TracingCond(true_fn, false_fn)
    cond = torch.tensor(True, dtype=torch.bool)
    x = torch.tensor([0, 1], dtype=torch.int)
    y = torch.tensor([2.0, 2.5])
    x1, y1 = if_else.cond(cond, x, y)
    print(x1, y1)
    trace_cond = jit(use_state(lambda: if_else.cond), trace=True, lazy=False, example_inputs=(cond, x, y))
    x1, y1 = trace_cond(cond, x, y)
    print(x1, y1)
    
    cond = torch.tensor([True, False, True], dtype=torch.bool)
    x = torch.tensor([0, 1, 2], dtype=torch.int)
    y = torch.tensor([[2.0, 2.5], [3.0, 3.5], [4.0, 4.5]])
    vmap_cond = jit(vmap(use_state(lambda: if_else.cond)), trace=True, lazy=False, example_inputs=(cond, x, y))
    x1, y1 = vmap_cond(cond, x, y)
    print(x1, y1)
    
    cond = torch.tensor([[True, False, True], [False, True, True]], dtype=torch.bool)
    x = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int)
    y = torch.tensor([[[2.0, 2.5], [3.0, 3.5], [4.0, 4.5]], [[2.1, 2.2], [3.1, 3.2], [4.1, 4.2]]])
    vmap_cond = jit(vmap(vmap(use_state(lambda: if_else.cond))), trace=True, lazy=False, example_inputs=(cond, x, y))
    x1, y1 = vmap_cond(cond, x, y)
    print(x1, y1)