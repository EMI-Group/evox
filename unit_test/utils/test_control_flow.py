import unittest
from typing import List, Tuple
import torch

from evox.core import ModuleBase, jit, jit_class, trace_impl, use_state, vmap
from evox.utils import TracingCond, TracingSwitch, TracingWhile


@jit_class
class MyModule(ModuleBase):
    def test(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.flatten()[0] > 0:
            return x + y
        else:
            return x * y

    @trace_impl(test)
    def trace_test(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        switch = TracingSwitch(self.true_branch, self.false_branch)
        return switch.switch((x.flatten()[0] > 0).to(dtype=torch.int), x, y)

    def true_branch(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y

    def false_branch(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x * y


class TestControlFlow(unittest.TestCase):

    def setUp(self):
        self.loop_body = lambda x, y: (x + 1, y**1.05)
        self.loop_cond = lambda x, y: x < 10
        self.while_loop = TracingWhile(self.loop_cond, self.loop_body)

        self.true_fn = lambda x, y: [x + 1, y**1.05]
        self.false_fn = lambda x, y: [x - 1, y**0.95]
        self.if_else = TracingCond(self.true_fn, self.false_fn)

    def test_while_loop(self):
        x = torch.tensor(0, dtype=torch.int)
        y = torch.tensor([2.0, 2.5])
        x1, y1 = self.while_loop.loop(x, y)
        self.assertTrue(torch.equal(x1, torch.tensor(10)))
        self.assertTrue(
            torch.allclose(y1, torch.tensor([2.0**1.05**10, 2.5**1.05**10]))
        )

    def test_jit_while_loop(self):
        x = torch.tensor(0, dtype=torch.int)
        y = torch.tensor([2.0, 2.5])
        trace_loop = jit(
            use_state(lambda: self.while_loop.loop),
            trace=True,
            lazy=False,
            example_inputs=(x, y),
        )
        x1, y1 = trace_loop(x, y)
        self.assertTrue(torch.equal(x1, torch.tensor(10)))
        self.assertTrue(
            torch.allclose(y1, torch.tensor([2.0**1.05**10, 2.5**1.05**10]))
        )

    def test_vmap_while_loop(self):
        x = torch.tensor([0, 1, 2], dtype=torch.int)
        y = torch.tensor([[2.0, 2.5], [3.0, 3.5], [4.0, 4.5]])
        vmap_loop = jit(
            vmap(use_state(lambda: self.while_loop.loop)),
            trace=True,
            lazy=False,
            example_inputs=(x, y),
        )
        x1, y1 = vmap_loop(x, y)
        self.assertTrue(torch.equal(x1, torch.tensor([10, 10, 10])))
        self.assertTrue(
            torch.allclose(
                y1,
                torch.tensor(
                    [
                        [2.0**1.05**10, 2.5**1.05**10],
                        [3.0**1.05**9, 3.5**1.05**9],
                        [4.0**1.05**8, 4.5**1.05**8],
                    ]
                ),
            )
        )

    def test_nested_vmap_while_loop(self):
        x = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.int)
        y = torch.tensor(
            [[[2.0, 2.5], [3.0, 3.5], [4.0, 4.5]], [[2.1, 2.2], [3.1, 3.2], [4.1, 4.2]]]
        )
        vmap_loop = jit(
            vmap(vmap(use_state(lambda: self.while_loop.loop))),
            trace=True,
            lazy=False,
            example_inputs=(x, y),
        )
        x1, y1 = vmap_loop(x, y)
        self.assertTrue(torch.equal(x1, torch.tensor([[10, 10, 10], [10, 10, 10]])))
        self.assertTrue(
            torch.allclose(
                y1,
                torch.tensor(
                    [
                        [
                            [2.0**1.05**10, 2.5**1.05**10],
                            [3.0**1.05**9, 3.5**1.05**9],
                            [4.0**1.05**8, 4.5**1.05**8],
                        ],
                        [
                            [2.1**1.05**10, 2.2**1.05**10],
                            [3.1**1.05**9, 3.2**1.05**9],
                            [4.1**1.05**8, 4.2**1.05**8],
                        ],
                    ]
                ),
            )
        )

    def test_if_else(self):
        cond = torch.tensor(True, dtype=torch.bool)
        x = torch.tensor([0, 1], dtype=torch.int)
        y = torch.tensor([2.0, 2.5])
        x1, y1 = self.if_else.cond(cond, x, y)
        self.assertTrue(torch.equal(x1, torch.tensor([1, 2])))
        self.assertTrue(torch.allclose(y1, torch.tensor([2.0**1.05, 2.5**1.05])))

    def test_jit_if_else(self):
        cond = torch.tensor(True, dtype=torch.bool)
        x = torch.tensor([0, 1], dtype=torch.int)
        y = torch.tensor([2.0, 2.5])
        trace_cond = jit(
            use_state(lambda: self.if_else.cond),
            trace=True,
            lazy=False,
            example_inputs=(cond, x, y),
        )
        x1, y1 = trace_cond(cond, x, y)
        self.assertTrue(torch.equal(x1, torch.tensor([1, 2])))
        self.assertTrue(torch.allclose(y1, torch.tensor([2.0**1.05, 2.5**1.05])))

    def test_vmap_if_else(self):
        cond = torch.tensor([True, False, True], dtype=torch.bool)
        x = torch.tensor([0, 1, 2], dtype=torch.int)
        y = torch.tensor([[2.0, 2.5], [3.0, 3.5], [4.0, 4.5]])
        vmap_cond = jit(
            vmap(use_state(lambda: self.if_else.cond)),
            trace=True,
            lazy=False,
            example_inputs=(cond, x, y),
        )
        x1, y1 = vmap_cond(cond, x, y)
        self.assertTrue(torch.equal(x1, torch.tensor([1, 0, 3])))
        self.assertTrue(
            torch.allclose(
                y1,
                torch.tensor(
                    [
                        [2.0**1.05, 2.5**1.05],
                        [3.0**0.95, 3.5**0.95],
                        [4.0**1.05, 4.5**1.05],
                    ]
                ),
            )
        )

    def test_nested_vmap_if_else(self):
        cond = torch.tensor(
            [[True, False, True], [False, True, True]], dtype=torch.bool
        )
        x = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int)
        y = torch.tensor(
            [[[2.0, 2.5], [3.0, 3.5], [4.0, 4.5]], [[2.1, 2.2], [3.1, 3.2], [4.1, 4.2]]]
        )
        vmap_cond = jit(
            vmap(vmap(use_state(lambda: self.if_else.cond))),
            trace=True,
            lazy=False,
            example_inputs=(cond, x, y),
        )
        x1, y1 = vmap_cond(cond, x, y)
        self.assertTrue(torch.equal(x1, torch.tensor([[1, 0, 3], [2, 5, 6]])))
        self.assertTrue(
            torch.allclose(
                y1,
                torch.tensor(
                    [
                        [
                            [2.0**1.05, 2.5**1.05],
                            [3.0**0.95, 3.5**0.95],
                            [4.0**1.05, 4.5**1.05],
                        ],
                        [
                            [2.1**0.95, 2.2**0.95],
                            [3.1**1.05, 3.2**1.05],
                            [4.1**1.05, 4.2**1.05],
                        ],
                    ]
                ),
            )
        )

    def test_jit_module_method(self):
        m = MyModule()
        x = torch.tensor([1, -1])
        y = torch.tensor([3, 4])
        self.assertTrue(torch.equal(m.test(x, y), torch.tensor([4, 3])))
        trace_cond = jit(
            use_state(lambda: m.test), trace=True, lazy=False, example_inputs=(x, y)
        )
        self.assertTrue(torch.equal(trace_cond(x, y), torch.tensor([3, -4])))
