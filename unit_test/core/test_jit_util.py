from functools import partial
import torch

import os
import sys
current_directory = os.getcwd()
if current_directory not in sys.path:
    sys.path.append(current_directory)

from src.core import vmap, jit


if __name__ == "__main__":

    @partial(vmap, example_ndim=2)
    def _single_eval(x: torch.Tensor, p: float = 2.0, q: torch.Tensor = torch.as_tensor(range(2))):
        return (x**p).sum() * q.sum()

    print(_single_eval(2 * torch.ones(10, 2)))
    print(jit(_single_eval)(2 * torch.ones(10, 2)))
    print(jit(_single_eval, trace=True, lazy=True)(2 * torch.ones(10, 2)))
