import torch
from core import vmap, Problem, Workflow, use_state, jit
# torch.manual_seed(42)
from functools import partial

def _single_eval(x: torch.Tensor, p: float = 2.0, q: torch.Tensor = torch.as_tensor(range(2))):
    return (x**p).sum() * q.sum()

_single_eval = vmap(_single_eval, example_ndim=2)

print(_single_eval(2 * torch.ones(10, 2)))
print(jit(_single_eval)(2 * torch.ones(10, 2)))
print(jit(_single_eval, trace=True, lazy=True)(2 * torch.ones(10, 2)))
# 生成随机数
# random_tensor = torch.rand(3, 3)
# print(random_tensor)
