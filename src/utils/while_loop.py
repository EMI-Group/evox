from typing import Callable, Dict, List, Tuple

import torch

from ..core import trace_impl, vmap_impl, jit_class, vmap, jit, ModuleBase
from ..core import _vmap_fix


@jit_class
class TracingWhileLoop(ModuleBase):
    """A helper class used to trace a while-loop.

    ## Usage
    ```
    def loop_body(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return x + 1, y ** 1.05

    def loop_cond(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x < 10

    while_loop = TracingWhileLoop(loop_cond, loop_body)
    # normal usage
    x = torch.tensor(0, dtype=torch.int)
    y = torch.tensor([2.0, 2.5])
    x1, y1 = while_loop.loop(x, y)
    print(x1, y1)
    # trace a while loop
    trace_loop = jit(use_state(lambda: while_loop.loop), trace=True, example_inputs=(x, y))
    x1, y1 = trace_loop(x, y)
    print(x1, y1)
    # vector-map a while loop
    x = torch.tensor([0, 1, 2], dtype=torch.int)
    y = torch.tensor([[2.0, 2.5], [3.0, 3.5], [4.0, 4.5]])
    vmap_loop = jit(vmap(use_state(lambda: while_loop.loop)), trace=True, example_inputs=(x, y))
    x1, y1 = vmap_loop(x, y)
    print(x1, y1)
    ```
    """

    def __init__(
        self,
        cond: Callable[[*Tuple[torch.Tensor, ...]], torch.Tensor],
        body: Callable[[*Tuple[torch.Tensor, ...]], Tuple[torch.Tensor, ...]],
    ):
        """
        Initialize the `TracingWhileLoop`.

        Args:
            cond (*torch.Tensor -> torch.Tensor): The condition function. Must be JIT-script compatible.
            body (*torch.Tensor -> *torch.Tensor): The body function. Must be JIT-script compatible.

        ## Notice:
        When using `TracingWhileLoop` and tracing JIT (`core.jit` with `trace=True`), the outer-most `core.jit` must have optional arguments `lazy=False` and `no_cache=False`.
        """
        super().__init__()
        self.cond = torch.jit.script(cond)
        self.body = torch.jit.script(body)
        self._cond = cond
        self._body = body
        self._cache_compiled_vmap_loop: Dict[Tuple[Tuple[int, ...], int], torch.jit.ScriptFunction] = {}

    @torch.jit.ignore
    def loop(self, *x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Executes a while-loop with the given condition and body functions.

        When tracing JIT (`core.jit` with `trace=True`), the `trace_loop` function is used instead; when using `core.vmap`, the `vmap_loop` function is used instead.

        Args:
            *x (`torch.Tensor`): The input tensors / carry for the loop.

        Returns:
            `Tuple[torch.Tensor, ...]`: The resulting tensors / carry after the loop completes.
        """
        while self.cond(*x):
            x = self.body(*x)
        return x

    @trace_impl(loop)
    def trace_loop(self, *x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        cond: Callable[[*Tuple[torch.Tensor, ...]], torch.Tensor] = self.cond
        body: Callable[[*Tuple[torch.Tensor, ...]], Tuple[torch.Tensor, ...]] = self.body

        def _loop1(x1: torch.Tensor):
            while cond(x1):
                x1 = body(x1)
            return x1

        def _loop2(x1: torch.Tensor, x2: torch.Tensor):
            while cond(x1, x2):
                x1, x2 = body(x1, x2)
            return x1, x2

        def _loop3(x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor):
            while cond(x1, x2, x3):
                x1, x2, x3 = body(x1, x2, x3)
            return x1, x2, x3

        def _loop4(x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, x4: torch.Tensor):
            while cond(x1, x2, x3, x4):
                x1, x2, x3, x4 = body(x1, x2, x3, x4)
            return x1, x2, x3, x4

        def _loop5(
            x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, x4: torch.Tensor, x5: torch.Tensor
        ):
            while cond(x1, x2, x3, x4, x5):
                x1, x2, x3, x4, x5 = body(x1, x2, x3, x4, x5)
            return x1, x2, x3, x4, x5

        def _loop6(
            x1: torch.Tensor,
            x2: torch.Tensor,
            x3: torch.Tensor,
            x4: torch.Tensor,
            x5: torch.Tensor,
            x6: torch.Tensor,
        ):
            while cond(x1, x2, x3, x4, x5, x6):
                x1, x2, x3, x4, x5, x6 = body(x1, x2, x3, x4, x5, x6)
            return x1, x2, x3, x4, x5, x6

        def _loop7(
            x1: torch.Tensor,
            x2: torch.Tensor,
            x3: torch.Tensor,
            x4: torch.Tensor,
            x5: torch.Tensor,
            x6: torch.Tensor,
            x7: torch.Tensor,
        ):
            while cond(x1, x2, x3, x4, x5, x6, x7):
                x1, x2, x3, x4, x5, x6, x7 = body(x1, x2, x3, x4, x5, x6, x7)
            return x1, x2, x3, x4, x5, x6, x7

        def _loop8(
            x1: torch.Tensor,
            x2: torch.Tensor,
            x3: torch.Tensor,
            x4: torch.Tensor,
            x5: torch.Tensor,
            x6: torch.Tensor,
            x7: torch.Tensor,
            x8: torch.Tensor,
        ):
            while cond(x1, x2, x3, x4, x5, x6, x7, x8):
                x1, x2, x3, x4, x5, x6, x7, x8 = body(x1, x2, x3, x4, x5, x6, x7, x8)
            return x1, x2, x3, x4, x5, x6, x7, x8

        def _loop9(
            x1: torch.Tensor,
            x2: torch.Tensor,
            x3: torch.Tensor,
            x4: torch.Tensor,
            x5: torch.Tensor,
            x6: torch.Tensor,
            x7: torch.Tensor,
            x8: torch.Tensor,
            x9: torch.Tensor,
        ):
            while cond(x1, x2, x3, x4, x5, x6, x7, x8, x9):
                x1, x2, x3, x4, x5, x6, x7, x8, x9 = body(x1, x2, x3, x4, x5, x6, x7, x8, x9)
            return x1, x2, x3, x4, x5, x6, x7, x8, x9

        loops_dict = {
            1: _loop1,
            2: _loop2,
            3: _loop3,
            4: _loop4,
            5: _loop5,
            6: _loop6,
            7: _loop7,
            8: _loop8,
            9: _loop9,
        }

        assert len(x) <= len(
            loops_dict
        ), f"At most {len(loops_dict)} arguments are supported, got {len(x)}"
        compiled_loop = torch.jit.script(loops_dict[len(x)])
        return compiled_loop(*x)

    @torch.jit.ignore
    def _compile_vmap_loop_fn(
        self, original_args: Tuple[torch.Tensor, ...], vmap_dims: Tuple[Tuple[int, ...], ...]
    ):
        # vmap
        vmap_cond = self._cond
        vmap_body = self._body
        for d in zip(*vmap_dims):
            vmap_cond = vmap(vmap_cond, in_dims=d, trace=False)
            vmap_body = vmap(vmap_body, in_dims=d, out_dims=d, trace=False)

        def _expand_vmap_dim(
            vmap_dim: Tuple[int, ...], size: Tuple[int, ...], a: torch.Tensor
        ) -> torch.Tensor:
            vmap_dim = tuple(d + i for i, d in enumerate(vmap_dim))
            size = list(size)
            for i, _ in enumerate(size):
                if i not in vmap_dim:
                    size[i] = 1
            return a.view(*size)

        def _vmap_cond_fn(*xs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
            cond_res = vmap_cond(*xs)
            cond_res = tuple(_expand_vmap_dim(d, a.size(), cond_res) for d, a in zip(vmap_dims, xs))
            return cond_res

        # JIT
        body = jit(vmap_body, trace=True, example_inputs=original_args)
        cond = jit(_vmap_cond_fn, trace=True, example_inputs=original_args)

        def _loop1(x1: torch.Tensor):
            cond_res = cond(x1)[0]
            while cond_res.any():
                x1_new = body(x1)
                x1 = torch.where(cond_res, x1_new, x1)
                cond_res = cond(x1)
            return x1

        def _loop2(x1: torch.Tensor, x2: torch.Tensor):
            cond_res1, cond_res2 = cond(x1, x2)
            while cond_res1.any():
                x1_new, x2_new = body(x1, x2)
                x1 = torch.where(cond_res1, x1_new, x1)
                x2 = torch.where(cond_res2, x2_new, x2)
                cond_res1, cond_res2 = cond(x1, x2)
            return x1, x2

        def _loop3(x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor):
            cond_res1, cond_res2, cond_res3 = cond(x1, x2, x3)
            while cond_res1.any():
                x1_new, x2_new, x3_new = body(x1, x2, x3)
                x1 = torch.where(cond_res1, x1_new, x1)
                x2 = torch.where(cond_res2, x2_new, x2)
                x3 = torch.where(cond_res3, x3_new, x3)
                cond_res1, cond_res2, cond_res3 = cond(x1, x2, x3)
            return x1, x2, x3

        def _loop4(x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, x4: torch.Tensor):
            cond_res1, cond_res2, cond_res3, cond_res4 = cond(x1, x2, x3, x4)
            while cond_res1.any():
                x1_new, x2_new, x3_new, x4_new = body(x1, x2, x3, x4)
                x1 = torch.where(cond_res1, x1_new, x1)
                x2 = torch.where(cond_res2, x2_new, x2)
                x3 = torch.where(cond_res3, x3_new, x3)
                x4 = torch.where(cond_res4, x4_new, x4)
                cond_res1, cond_res2, cond_res3, cond_res4 = cond(x1, x2, x3, x4)
            return x1, x2, x3, x4

        def _loop5(
            x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, x4: torch.Tensor, x5: torch.Tensor
        ):
            cond_res1, cond_res2, cond_res3, cond_res4, cond_res5 = cond(x1, x2, x3, x4, x5)
            while cond_res1.any():
                x1_new, x2_new, x3_new, x4_new, x5_new = body(x1, x2, x3, x4, x5)
                x1 = torch.where(cond_res1, x1_new, x1)
                x2 = torch.where(cond_res2, x2_new, x2)
                x3 = torch.where(cond_res3, x3_new, x3)
                x4 = torch.where(cond_res4, x4_new, x4)
                x5 = torch.where(cond_res5, x5_new, x5)
                cond_res1, cond_res2, cond_res3, cond_res4, cond_res5 = cond(x1, x2, x3, x4, x5)
            return x1, x2, x3, x4, x5

        def _loop6(
            x1: torch.Tensor,
            x2: torch.Tensor,
            x3: torch.Tensor,
            x4: torch.Tensor,
            x5: torch.Tensor,
            x6: torch.Tensor,
        ):
            cond_res1, cond_res2, cond_res3, cond_res4, cond_res5, cond_res6 = cond(
                x1, x2, x3, x4, x5, x6
            )
            while cond_res1.any():
                x1_new, x2_new, x3_new, x4_new, x5_new, x6_new = body(x1, x2, x3, x4, x5, x6)
                x1 = torch.where(cond_res1, x1_new, x1)
                x2 = torch.where(cond_res2, x2_new, x2)
                x3 = torch.where(cond_res3, x3_new, x3)
                x4 = torch.where(cond_res4, x4_new, x4)
                x5 = torch.where(cond_res5, x5_new, x5)
                x6 = torch.where(cond_res6, x6_new, x6)
                cond_res1, cond_res2, cond_res3, cond_res4, cond_res5, cond_res6 = cond(
                    x1, x2, x3, x4, x5, x6
                )
            return x1, x2, x3, x4, x5, x6

        def _loop7(
            x1: torch.Tensor,
            x2: torch.Tensor,
            x3: torch.Tensor,
            x4: torch.Tensor,
            x5: torch.Tensor,
            x6: torch.Tensor,
            x7: torch.Tensor,
        ):
            cond_res1, cond_res2, cond_res3, cond_res4, cond_res5, cond_res6, cond_res7 = cond(
                x1, x2, x3, x4, x5, x6, x7
            )
            while cond_res1.any():
                x1_new, x2_new, x3_new, x4_new, x5_new, x6_new, x7_new = body(x1, x2, x3, x4, x5, x6, x7)
                x1 = torch.where(cond_res1, x1_new, x1)
                x2 = torch.where(cond_res2, x2_new, x2)
                x3 = torch.where(cond_res3, x3_new, x3)
                x4 = torch.where(cond_res4, x4_new, x4)
                x5 = torch.where(cond_res5, x5_new, x5)
                x6 = torch.where(cond_res6, x6_new, x6)
                x7 = torch.where(cond_res7, x7_new, x7)
                cond_res1, cond_res2, cond_res3, cond_res4, cond_res5, cond_res6, cond_res7 = cond(
                    x1, x2, x3, x4, x5, x6, x7
                )
            return x1, x2, x3, x4, x5, x6, x7

        def _loop8(
            x1: torch.Tensor,
            x2: torch.Tensor,
            x3: torch.Tensor,
            x4: torch.Tensor,
            x5: torch.Tensor,
            x6: torch.Tensor,
            x7: torch.Tensor,
            x8: torch.Tensor,
        ):
            cond_res1, cond_res2, cond_res3, cond_res4, cond_res5, cond_res6, cond_res7, cond_res8 = (
                cond(x1, x2, x3, x4, x5, x6, x7, x8)
            )
            while cond_res1.any():
                x1_new, x2_new, x3_new, x4_new, x5_new, x6_new, x7_new, x8_new = body(
                    x1, x2, x3, x4, x5, x6, x7, x8
                )
                x1 = torch.where(cond_res1, x1_new, x1)
                x2 = torch.where(cond_res2, x2_new, x2)
                x3 = torch.where(cond_res3, x3_new, x3)
                x4 = torch.where(cond_res4, x4_new, x4)
                x5 = torch.where(cond_res5, x5_new, x5)
                x6 = torch.where(cond_res6, x6_new, x6)
                x7 = torch.where(cond_res7, x7_new, x7)
                x8 = torch.where(cond_res8, x8_new, x8)
                (
                    cond_res1,
                    cond_res2,
                    cond_res3,
                    cond_res4,
                    cond_res5,
                    cond_res6,
                    cond_res7,
                    cond_res8,
                ) = cond(x1, x2, x3, x4, x5, x6, x7, x8)
            return x1, x2, x3, x4, x5, x6, x7, x8

        def _loop9(
            x1: torch.Tensor,
            x2: torch.Tensor,
            x3: torch.Tensor,
            x4: torch.Tensor,
            x5: torch.Tensor,
            x6: torch.Tensor,
            x7: torch.Tensor,
            x8: torch.Tensor,
            x9: torch.Tensor,
        ):
            (
                cond_res1,
                cond_res2,
                cond_res3,
                cond_res4,
                cond_res5,
                cond_res6,
                cond_res7,
                cond_res8,
                cond_res9,
            ) = cond(x1, x2, x3, x4, x5, x6, x7, x8, x9)
            while cond_res1.any():
                x1_new, x2_new, x3_new, x4_new, x5_new, x6_new, x7_new, x8_new, x9_new = body(
                    x1, x2, x3, x4, x5, x6, x7, x8, x9
                )
                x1 = torch.where(cond_res1, x1_new, x1)
                x2 = torch.where(cond_res2, x2_new, x2)
                x3 = torch.where(cond_res3, x3_new, x3)
                x4 = torch.where(cond_res4, x4_new, x4)
                x5 = torch.where(cond_res5, x5_new, x5)
                x6 = torch.where(cond_res6, x6_new, x6)
                x7 = torch.where(cond_res7, x7_new, x7)
                x8 = torch.where(cond_res8, x8_new, x8)
                x9 = torch.where(cond_res9, x9_new, x9)
                (
                    cond_res1,
                    cond_res2,
                    cond_res3,
                    cond_res4,
                    cond_res5,
                    cond_res6,
                    cond_res7,
                    cond_res8,
                    cond_res9,
                ) = cond(x1, x2, x3, x4, x5, x6, x7, x8, x9)
            return x1, x2, x3, x4, x5, x6, x7, x8, x9

        loops_dict = {
            1: _loop1,
            2: _loop2,
            3: _loop3,
            4: _loop4,
            5: _loop5,
            6: _loop6,
            7: _loop7,
            8: _loop8,
            9: _loop9,
        }
        assert len(original_args) <= len(
            loops_dict
        ), f"At most {len(loops_dict)} arguments are supported, got {len(original_args)}"
        return torch.jit.script(loops_dict[len(original_args)])

    @vmap_impl(loop)
    def vmap_loop(self, *x: torch.Tensor) -> torch.Tensor:
        # get vmap dims and original arguments
        vmap_dims = []
        original_args: List[torch.Tensor] = []
        for arg in x:
            assert isinstance(
                arg, torch.Tensor
            ), f"Expect all arguments in `vmap` to be `torch.Tensor`, got {type(arg)}"
            arg, in_dim, _ = _vmap_fix.unwrap_batch_tensor(arg)
            vmap_dims.append(in_dim)
            original_args.append(arg)
        original_args = tuple(original_args)
        # compile
        key = tuple((d, a.ndim) for d, a in zip(vmap_dims, original_args))
        if key in self._cache_compiled_vmap_loop:
            vmap_loop_compiled = self._cache_compiled_vmap_loop[key]
        else:
            vmap_loop_compiled = self._compile_vmap_loop_fn(original_args, vmap_dims)
            self._cache_compiled_vmap_loop[key] = vmap_loop_compiled
        ret = vmap_loop_compiled(*original_args)
        returns = []
        for r, d in zip(ret, vmap_dims):
            for level, dim in enumerate(d, 1):
                r = _vmap_fix.add_batch_dim(r, dim, level)
            returns.append(r)
        return returns
