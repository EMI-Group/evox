import unittest
from typing import Dict, List

import torch
import torch.nn as nn

from evox.core import ModuleBase, trace_impl, use_state


class DummyModule(ModuleBase):
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


class TestModule(unittest.TestCase):
    def setUp(self):
        self.test_instance = DummyModule()

    def test_h_function(self):
        q = torch.rand(100, 4)
        result = self.test_instance.h(q)
        self.assertIsInstance(result, torch.Tensor)

    def test_g_function(self):
        p = torch.rand(100, 4)
        result = self.test_instance.g(p)
        self.assertIsInstance(result, torch.Tensor)

    def test_th_function(self):
        q = torch.rand(100, 4)
        result = self.test_instance.th(q)
        self.assertIsInstance(result, torch.Tensor)

    def test_add_mutable_list(self):
        self.test_instance.add_mutable("mut_list", [torch.zeros(10), torch.ones(10)])
        self.assertTrue(torch.equal(self.test_instance.mut_list[0], torch.zeros(10)))
        self.assertTrue(torch.equal(self.test_instance.mut_list[1], torch.ones(10)))

    def test_add_mutable_dict(self):
        self.test_instance.add_mutable("mut_dict", {"a": torch.zeros(20), "b": torch.ones(20)})
        self.assertTrue(torch.equal(self.test_instance.mut_dict["a"], torch.zeros(20)))
        self.assertTrue(torch.equal(self.test_instance.mut_dict["b"], torch.ones(20)))

    def test_trace_fn(self):
        fn = use_state(lambda: self.test_instance.h, is_generator=True)
        trace_fn = torch.jit.trace(fn, (fn.init_state(), torch.ones(10, 1)), strict=False)
        self.assertIsNotNone(trace_fn)

    def test_loop_function(self):
        fn = use_state(lambda: self.test_instance.h, is_generator=True)
        trace_fn = torch.jit.trace(fn, (fn.init_state(), torch.ones(10, 1)), strict=False)

        def loop(init_state: Dict[str, torch.Tensor], init_x: torch.Tensor, n: int = 10):
            state = init_state
            ret = init_x
            rets: List[torch.Tensor] = []
            for _ in range(n):
                state, ret = trace_fn(state, ret)
                rets.append(state["self.sub_mod.buf"])
            return rets

        loop_traced = torch.jit.trace(loop, (fn.init_state(), torch.rand(10, 2)), strict=False)
        self.assertIsNotNone(loop_traced)
