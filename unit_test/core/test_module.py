import torch
import torch.nn as nn
from typing import Dict, List

import os
import sys
current_directory = os.getcwd()
if current_directory not in sys.path:
    sys.path.append(current_directory)

from src.core import jit_class, ModuleBase, trace_impl, use_state, jit


if __name__ == "__main__":

    @jit_class
    class Test(ModuleBase):

        def __init__(self, threshold=0.5):
            super().__init__()
            self.threshold = threshold
            self.sub_mod = nn.Module()
            self.sub_mod.buf = nn.Buffer(torch.zeros(()))

        def h(self, q: torch.Tensor) -> torch.Tensor:
            if q.flatten()[0] > self.threshold:
                x = torch.sin(q)
            else:
                x = torch.tan(q)
            return x * x.shape[1]

        @trace_impl(h)
        def th(self, q: torch.Tensor) -> torch.Tensor:
            x = torch.where(q.flatten()[0] > self.threshold, q + 2, q + 5)
            x += self.g(x).abs()
            x *= x.shape[1]
            self.sub_mod.buf = x.sum()
            return x

        def g(self, p: torch.Tensor) -> torch.Tensor:
            x = torch.cos(p)
            return x * p.shape[0]

    t = Test()
    print(t.h.inlined_graph)
    result = t.g(torch.rand(100, 4))
    print(result)
    t.add_mutable("mut_list", [torch.zeros(10), torch.ones(10)])
    t.add_mutable("mut_dict", {"a": torch.zeros(20), "b": torch.ones(20)})
    print(t.mut_list[0])
    print(t.mut_dict["b"])

    t = Test()
    fn = use_state(lambda: t.h, is_generator=True)
    trace_fn = jit(fn, trace=True, lazy=False, example_inputs=(fn.init_state(), torch.ones(10, 1)))

    def loop(init_state: Dict[str, torch.Tensor], init_x: torch.Tensor, n: int = 10):
        state = init_state
        ret = init_x
        rets: List[torch.Tensor] = []
        for _ in range(n):
            state, ret = trace_fn(state, ret)
            rets.append(state["self.sub_mod.buf"])
        return rets

    print(trace_fn.code)
    loop = jit(loop, trace=True, lazy=False, example_inputs=(fn.init_state(), torch.rand(10, 2)))
    print(loop.code)
    print(loop(fn.init_state(), torch.rand(10, 2)))
