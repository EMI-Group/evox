import unittest

import torch

from evox.core import ModuleBase, Mutable, jit, jit_class, trace_impl, use_state, vmap
from evox.utils import TracingCond, TracingSwitch, TracingWhile


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
        self.assertTrue(torch.allclose(y1, torch.tensor([2.0**1.05**10, 2.5**1.05**10])))

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
        self.assertTrue(torch.allclose(y1, torch.tensor([2.0**1.05**10, 2.5**1.05**10])))

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
        y = torch.tensor([[[2.0, 2.5], [3.0, 3.5], [4.0, 4.5]], [[2.1, 2.2], [3.1, 3.2], [4.1, 4.2]]])
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
        cond = torch.tensor([[True, False, True], [False, True, True]], dtype=torch.bool)
        x = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int)
        y = torch.tensor([[[2.0, 2.5], [3.0, 3.5], [4.0, 4.5]], [[2.1, 2.2], [3.1, 3.2], [4.1, 4.2]]])
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

    def test_jit_module_while_loop(self):
        @jit_class
        class WhileModule(ModuleBase):
            def __init__(self):
                super().__init__()
                self.iters = Mutable(torch.tensor(0, dtype=torch.int))

            def test(self, x: torch.Tensor, y: torch.Tensor):
                while x.flatten()[0] < 10:
                    x = x + y
                    y = y / 1.1
                    self.iters += 1
                return x, y

            @trace_impl(test)
            def trace_test(self, x: torch.Tensor, y: torch.Tensor):
                while_loop = TracingWhile(self.cond_fn, self.body_fn)
                return while_loop.loop(x, y)

            def cond_fn(self, x: torch.Tensor, y: torch.Tensor):
                return x.flatten()[0] < 10

            def body_fn(self, x: torch.Tensor, y: torch.Tensor):
                self.iters += 1
                return x + y, y / 1.1

        m = WhileModule()
        x = torch.tensor([1.0, -1.0])
        y = torch.tensor([3.0, 4.0])
        print(m.test(x, y))
        state_loop = use_state(lambda: m.test)
        trace_loop = jit(use_state(lambda: m.test), trace=True, lazy=False, example_inputs=(state_loop.init_state(False), x, y))
        print(trace_loop(state_loop.init_state(False), x, y))

    def test_jit_module_switch(self):

        @jit_class
        class SwitchModule(ModuleBase):
            def __init__(self):
                super().__init__()
                self.q = Mutable(torch.zeros(1))

            def test(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                if x.flatten()[0] > 0:
                    return x + y
                else:
                    return x * y

            @trace_impl(test)
            def trace_test(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                self.q = self.q + 1
                local_q = self.q * 2

                def false_branch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                    nonlocal local_q
                    local_q *= 1.5
                    return x * y * local_q

                ori_state, locals_vars = self.state_dict_and_nonlocal_vars(self.true_branch, false_branch)
                state = dict(ori_state)
                state.update(locals_vars)
                if not hasattr(self, "_switch_"):
                    self._switch_ = TracingSwitch(self.true_branch, false_branch)
                state, ret = self._switch_.switch(state, (x.flatten()[0] > 0).to(dtype=torch.int), x, y)
                locals_vars = {k: state[k] for k in locals_vars if k in state}
                ori_state = {k: state[k] for k in ori_state if k in state}
                self.load_state_dict(ori_state)
                for k in locals_vars:
                    exec(f"{k} = locals_vars['{k}']")
                return ret

            def true_branch(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                self.q = self.q + 1
                return x + y

        m = SwitchModule()
        x = torch.tensor([1, -1])
        y = torch.tensor([3, 4])
        self.assertTrue(torch.equal(m.test(x, y), torch.tensor([4, 3])))
        state_switch = use_state(lambda: m.test)
        trace_switch = jit(state_switch, trace=True, lazy=False, example_inputs=(state_switch.init_state(), x, y))
        self.assertTrue(torch.equal(trace_switch(state_switch.init_state(), x, y)[1], torch.tensor([3, -4]) * 2 * 1.5))


if __name__ == "__main__":
    test = TestControlFlow()
    test.setUp()
    test.test_jit_module_switch()
    test.test_jit_module_while_loop()
