import unittest

import torch
import torch.nn as nn

from evox.core import ModuleBase, use_state


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

    def g(self, p: torch.Tensor) -> torch.Tensor:
        x = torch.cos(p)
        return x * p.shape[0]

    def add(self, x: torch.Tensor):
        self.sub_mod.buf += x


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
        result = self.test_instance.h(q)
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
        fn = use_state(self.test_instance.h)
        state = (dict(self.test_instance.named_parameters()), dict(self.test_instance.named_buffers()))
        new_state, out = fn(state, torch.ones(3, 2))
        self.assertIsNotNone(new_state)
        self.assertIsNotNone(out)

    def test_inplace_op(self):
        self.test_instance.add(torch.ones(()))
        self.assertTrue(torch.equal(self.test_instance.sub_mod.buf, torch.ones(())))

    def test_use_state_inplace_op(self):
        fn = use_state(self.test_instance.add)
        state = (dict(self.test_instance.named_parameters()), dict(self.test_instance.named_buffers()))
        state = fn(state, torch.ones(()))
        params, buffers = state
        self.assertTrue(torch.equal(buffers["sub_mod.buf"], torch.ones(())))

        state = fn(state, torch.ones(()))
        params, buffers = state
        self.assertTrue(torch.equal(buffers["sub_mod.buf"], 2 * torch.ones(())))
