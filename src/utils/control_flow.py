from typing import Callable, Dict, List, Tuple, Iterable

import torch

from ..core import trace_impl, vmap_impl, jit_class, vmap, jit, use_state, ModuleBase
from ..core import _vmap_fix
from ..core.module import UseStateFunc


@jit_class
class TracingWhile(ModuleBase):
    """A helper class used to trace a while-loop.

    ## Usage
    ```
    def loop_body(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return x + 1, y ** 1.05

    def loop_cond(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x < 10

    while_loop = TracingWhile(loop_cond, loop_body)
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
        script_functions: bool = False,
    ):
        """
        Initialize the `TracingWhile`.

        Args:
            cond (`*torch.Tensor -> torch.Tensor`): The condition function. Must be JIT-script compatible if `script_functions=True`.
            body (`*torch.Tensor -> *torch.Tensor`): The body function. Must be JIT-script compatible if `script_functions=True`.
            script_functions (`bool`, optional): Whether the `cond` and `body` functions are JIT-script instantly. Defaults to False. When set to True, the basic `loop` function (outside JIT tracing or vector-map) may gain some performance improvement. However, it is not recommended to use `script_functions=True` since the basic `loop` function shall NOT be used in performance-critical paths.

        ## Notice:
        When using `TracingWhile` and tracing JIT (`core.jit` with `trace=True`), the outer-most `core.jit` must have optional arguments `lazy=False` and `no_cache=False`.
        """
        super().__init__()
        self.cond = torch.jit.script(cond) if script_functions else None
        self.body = torch.jit.script(body) if script_functions else None
        self._cond = cond
        self._body = body
        self._cache_compiled_vmap_loop: Dict[
            Tuple[Tuple[int, ...], int, torch.dtype, torch.device], torch.jit.ScriptFunction
        ] = {}
        self._cache_compiled_loop: Dict[
            Tuple[int, torch.dtype, torch.device], torch.jit.ScriptFunction
        ] = {}

    @torch.jit.ignore
    def loop(self, *x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Executes a while-loop with the given condition and body functions.

        When tracing JIT (`core.jit` with `trace=True`), the `trace_loop` function is used instead; when using `core.vmap`, the `vmap_loop` function is used instead.

        ## Notice:
        During normal `torch.jit.script`, this function shall NEVER be invoked for performance-critical paths, please use Python while loop directly.

        Args:
            *x (`torch.Tensor`): The input tensors / carry for the loop.

        Returns:
            `Tuple[torch.Tensor, ...]`: The resulting tensors / carry after the loop completes.
        """
        if self.cond is None or self.body is None:
            while self._cond(*x):
                x = self._body(*x)
        else:
            while self.cond(*x):
                x = self.body(*x)
        return x

    @torch.jit.ignore
    def _compile_loop_fn(self, original_args: Tuple[torch.Tensor, ...]) -> torch.jit.ScriptFunction:
        cond: Callable[[*Tuple[torch.Tensor, ...]], torch.Tensor] = self.cond
        body: Callable[[*Tuple[torch.Tensor, ...]], Tuple[torch.Tensor, ...]] = self.body
        if cond is None or body is None:
            cond = jit(self._cond, trace=True, lazy=False, example_inputs=original_args)
            body = jit(self._body, trace=True, lazy=False, example_inputs=original_args)

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

        assert len(original_args) <= len(
            loops_dict
        ), f"At most {len(loops_dict)} arguments are supported, got {len(original_args)}"
        compiled_loop = torch.jit.script(loops_dict[len(original_args)])
        return compiled_loop

    @trace_impl(loop)
    def trace_loop(self, *x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        key = tuple((a.ndim, a.dtype, a.device) for a in x)
        if key in self._cache_compiled_loop:
            compiled_loop = self._cache_compiled_loop[key]
        else:
            compiled_loop = self._compile_loop_fn(x)
            self._cache_compiled_loop[key] = compiled_loop
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
        key = tuple((d, a.ndim, a.dtype, a.device) for d, a in zip(vmap_dims, original_args))
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


@jit_class
class TracingCond(ModuleBase):
    """A helper class used to trace an if-else control flow.

    ## Usage
    ```
    def true_fn(x: torch.Tensor, y: torch.Tensor) -> List[torch.Tensor]:
        return [x + 1, y ** 1.05]

    def false_fn(x: torch.Tensor, y: torch.Tensor) -> List[torch.Tensor]:
        return [x - 1, y ** 0.95]

    if_else = TracingCond(true_fn, false_fn)
    # normal usage
    cond = torch.tensor(True, dtype=torch.bool)
    x = torch.tensor([0, 1], dtype=torch.int)
    y = torch.tensor([2.0, 2.5])
    x1, y1 = if_else.cond(cond, x, y)
    print(x1, y1)
    # trace a condition
    trace_cond = jit(use_state(lambda: if_else.cond), trace=True, lazy=False, example_inputs=(cond, x, y))
    x1, y1 = trace_cond(cond, x, y)
    print(x1, y1)
    # vmap a condition
    cond = torch.tensor([True, False, True], dtype=torch.bool)
    x = torch.tensor([0, 1, 2], dtype=torch.int)
    y = torch.tensor([[2.0, 2.5], [3.0, 3.5], [4.0, 4.5]])
    vmap_cond = jit(vmap(use_state(lambda: if_else.cond)), trace=True, lazy=False, example_inputs=(cond, x, y))
    x1, y1 = vmap_cond(cond, x, y)
    print(x1, y1)
    ```
    """

    def __init__(
        self,
        true_fn: Callable[[*Tuple[torch.Tensor, ...]], List[torch.Tensor]],
        false_fn: Callable[[*Tuple[torch.Tensor, ...]], List[torch.Tensor]],
        script_functions: bool = False,
    ):
        """
        Initialize the `TracingCond`.

        Args:
            true_fn (`*torch.Tensor -> *torch.Tensor`): The true branch function. Must be JIT-script compatible if `script_functions=True`.
            false_fn (`*torch.Tensor -> *torch.Tensor`): The false branch function. Must be JIT-script compatible if `script_functions=True`.
            script_functions (`bool`, optional): Whether the `true_fn` and `false_fn` functions are JIT-script instantly. Defaults to False. When set to True, the basic `cond` function (outside JIT tracing or vector-map) may gain some performance improvement. However, it is not recommended to use `script_functions=True` since the basic `cond` function shall NOT be used in performance-critical paths.
        """
        super().__init__()
        self.true_fn = torch.jit.script(true_fn) if script_functions else None
        self.false_fn = torch.jit.script(false_fn) if script_functions else None
        self._true_fn = true_fn
        self._false_fn = false_fn
        self._cache_compiled_cond: Dict[
            Tuple[int, torch.dtype, torch.device],
            Tuple[torch.jit.ScriptFunction, UseStateFunc, UseStateFunc],
        ] = {}

    @torch.jit.ignore
    def cond(self, cond: torch.Tensor, *x: torch.Tensor) -> List[torch.Tensor]:
        """Runs either the true or false branch based on the given condition.

        When tracing JIT (`core.jit` with `trace=True`), the `trace_cond` function is used instead; when using `core.vmap`, the `vmap_cond` function is used instead.

        ## Notice:
        During normal `torch.jit.script`, this function shall NEVER be invoked for performance-critical paths, please use Python if-else directly.

        Args:
            cond (`torch.Tensor`): A boolean tensor. If `True`, the true branch is run; if `False`, the false branch is run.
            *x (`*torch.Tensor`): The input tensors to the branch functions.

        Returns:
            `List[torch.Tensor]`: The output tensors from the chosen branch function.
        """
        if self.true_fn is None or self.false_fn is None:
            if cond:
                x = self._true_fn(*x)
            else:
                x = self._false_fn(*x)
        else:
            if cond:
                x = self.true_fn(*x)
            else:
                x = self.false_fn(*x)
        return x

    @torch.jit.ignore
    def _compile_cond_fn(self, original_args: Tuple[torch.Tensor, ...]):
        # use state for in-place modifications
        state_true_fn = use_state(lambda: self._true_fn)
        state_false_fn = use_state(lambda: self._false_fn)
        true_fn = jit(
            state_true_fn,
            trace=True,
            lazy=False,
            example_inputs=(state_true_fn.init_state(),) + original_args,
        )
        false_fn = jit(
            state_false_fn,
            trace=True,
            lazy=False,
            example_inputs=(state_false_fn.init_state(),) + original_args,
        )

        def _cond1(
            state_T: Dict[str, torch.Tensor],
            state_F: Dict[str, torch.Tensor],
            condition: torch.Tensor,
            x1: torch.Tensor,
        ):
            if condition:
                x = true_fn(state_T, x1)
            else:
                x = false_fn(state_F, x1)
            return x

        def _cond2(
            state_T: Dict[str, torch.Tensor],
            state_F: Dict[str, torch.Tensor],
            condition: torch.Tensor,
            x1: torch.Tensor,
            x2: torch.Tensor,
        ):
            if condition:
                x = true_fn(state_T, x1, x2)
            else:
                x = false_fn(state_F, x1, x2)
            return x

        def _cond3(
            state_T: Dict[str, torch.Tensor],
            state_F: Dict[str, torch.Tensor],
            condition: torch.Tensor,
            x1: torch.Tensor,
            x2: torch.Tensor,
            x3: torch.Tensor,
        ):
            if condition:
                x = true_fn(state_T, x1, x2, x3)
            else:
                x = false_fn(state_F, x1, x2, x3)
            return x

        def _cond4(
            state_T: Dict[str, torch.Tensor],
            state_F: Dict[str, torch.Tensor],
            condition: torch.Tensor,
            x1: torch.Tensor,
            x2: torch.Tensor,
            x3: torch.Tensor,
            x4: torch.Tensor,
        ):
            if condition:
                x = true_fn(state_T, x1, x2, x3, x4)
            else:
                x = false_fn(state_F, x1, x2, x3, x4)
            return x

        def _cond5(
            state_T: Dict[str, torch.Tensor],
            state_F: Dict[str, torch.Tensor],
            condition: torch.Tensor,
            x1: torch.Tensor,
            x2: torch.Tensor,
            x3: torch.Tensor,
            x4: torch.Tensor,
            x5: torch.Tensor,
        ):
            if condition:
                x = true_fn(state_T, x1, x2, x3, x4, x5)
            else:
                x = false_fn(state_F, x1, x2, x3, x4, x5)
            return x

        def _cond6(
            state_T: Dict[str, torch.Tensor],
            state_F: Dict[str, torch.Tensor],
            condition: torch.Tensor,
            x1: torch.Tensor,
            x2: torch.Tensor,
            x3: torch.Tensor,
            x4: torch.Tensor,
            x5: torch.Tensor,
            x6: torch.Tensor,
        ):
            if condition:
                x = true_fn(state_T, x1, x2, x3, x4, x5, x6)
            else:
                x = false_fn(state_F, x1, x2, x3, x4, x5, x6)
            return x

        def _cond7(
            state_T: Dict[str, torch.Tensor],
            state_F: Dict[str, torch.Tensor],
            condition: torch.Tensor,
            x1: torch.Tensor,
            x2: torch.Tensor,
            x3: torch.Tensor,
            x4: torch.Tensor,
            x5: torch.Tensor,
            x6: torch.Tensor,
            x7: torch.Tensor,
        ):
            if condition:
                x = true_fn(state_T, x1, x2, x3, x4, x5, x6, x7)
            else:
                x = false_fn(state_F, x1, x2, x3, x4, x5, x6, x7)
            return x

        def _cond8(
            state_T: Dict[str, torch.Tensor],
            state_F: Dict[str, torch.Tensor],
            condition: torch.Tensor,
            x1: torch.Tensor,
            x2: torch.Tensor,
            x3: torch.Tensor,
            x4: torch.Tensor,
            x5: torch.Tensor,
            x6: torch.Tensor,
            x7: torch.Tensor,
            x8: torch.Tensor,
        ):
            if condition:
                x = true_fn(state_T, x1, x2, x3, x4, x5, x6, x7, x8)
            else:
                x = false_fn(state_F, x1, x2, x3, x4, x5, x6, x7, x8)
            return x

        def _cond9(
            state_T: Dict[str, torch.Tensor],
            state_F: Dict[str, torch.Tensor],
            condition: torch.Tensor,
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
            if condition:
                x = true_fn(state_T, x1, x2, x3, x4, x5, x6, x7, x8, x9)
            else:
                x = false_fn(state_F, x1, x2, x3, x4, x5, x6, x7, x8, x9)
            return x

        cond_dict = {
            1: _cond1,
            2: _cond2,
            3: _cond3,
            4: _cond4,
            5: _cond5,
            6: _cond6,
            7: _cond7,
            8: _cond8,
            9: _cond9,
        }
        assert len(original_args) <= len(
            cond_dict
        ), f"At most {len(cond_dict)} arguments are supported, got {len(original_args)}"
        compiled_cond = torch.jit.script(cond_dict[len(original_args)])
        return compiled_cond, state_true_fn, state_false_fn

    @trace_impl(cond)
    def trace_cond(self, cond: torch.Tensor, *x: torch.Tensor) -> List[torch.Tensor]:
        key = tuple((a.ndim, a.dtype, a.device) for a in (cond,) + x)
        if key in self._cache_compiled_cond:
            compiled_cond, state_true_fn, state_false_fn = self._cache_compiled_cond[key]
        else:
            compiled_cond, state_true_fn, state_false_fn = self._compile_cond_fn(x)
            self._cache_compiled_cond[key] = (compiled_cond, state_true_fn, state_false_fn)
        res = compiled_cond(state_true_fn.init_state(False), state_false_fn.init_state(False), cond, *x)
        if isinstance(res, tuple):
            state, res = res
        else:
            state = res
            res = None
        state_true_fn.set_state(state)
        state_false_fn.set_state(state)
        return res

    @vmap_impl(cond)
    def vmap_cond(self, cond: torch.Tensor, *x: torch.Tensor) -> List[torch.Tensor]:
        # cannot dynamic dispatch for vmap, change to torch.where
        # use_state to support in-place modifications in true_fn and false_fn
        state_true_fn = use_state(lambda: self._true_fn)
        state_false_fn = use_state(lambda: self._false_fn)
        init_state = state_true_fn.init_state()
        init_state.update(state_false_fn.init_state())
        true_res = state_true_fn(state_true_fn.init_state(), *x)
        false_res = state_false_fn(state_false_fn.init_state(), *x)
        # unwrap state
        if isinstance(true_res, tuple):
            state_true, true_res = true_res
        else:
            state_true = true_res
            true_res = None
        if isinstance(false_res, tuple):
            state_false, false_res = false_res
        else:
            state_false = false_res
            false_res = None
        # get conditional outputs
        if isinstance(true_res, Iterable) and isinstance(false_res, Iterable):
            res = []
            for t, f in zip(true_res, false_res):
                res.append(torch.where(cond, t, f))
        elif true_res is None and false_res is None:
            res = None
        elif not isinstance(true_res, Iterable) and not isinstance(false_res, Iterable):
            res = torch.where(cond, true_res, false_res)
        else:
            raise ValueError("The type of returns of true_fn and false_fn should be the same.")
        # set conditional state
        state_out = {}
        for k in set(state_true.keys()).union(set(state_false.keys())):
            if k not in init_state:
                continue
            if state_true.get(k) is not None and state_false.get(k) is not None:
                state_out[k] = torch.where(cond, state_true[k], state_false[k])
            elif state_true.get(k) is not None:
                state_out[k] = torch.where(cond, state_true[k], init_state[k])
            elif state_false.get(k) is not None:
                state_out[k] = torch.where(cond, init_state[k], state_false[k])
        state_true_fn.set_state(state_out)
        state_false_fn.set_state(state_out)
        # return
        return res
