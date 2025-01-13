import inspect
import weakref
from typing import Dict, List, Tuple

import torch

from ...core import ModuleBase, _vmap_fix, jit, jit_class, trace_impl, use_state, vmap_impl
from ...core.module import UseStateFunc
from .utils import VarArgsCallable, VarArgsCallableMultiRet, _get_cache_key_object

_while_object_cache = weakref.WeakValueDictionary()


@jit_class
class TracingWhile(ModuleBase):
    """A helper class used to trace a while-loop."""

    def __new__(cls, cond_fn, body_fn, stateful_functions: bool | None = None):
        key_or_obj = _get_cache_key_object(_while_object_cache, cond_fn, body_fn)
        if isinstance(key_or_obj, tuple):
            obj = super().__new__(cls)
            obj.__cache_key__ = key_or_obj
            return obj
        else:
            return key_or_obj

    def __init__(self, cond_fn: VarArgsCallable, body_fn: VarArgsCallableMultiRet, stateful_functions: bool | None = None):
        """
        Initialize the `TracingWhile`.

        :param cond_fn: The condition function.
        :param body_fn: The body function.
        :param stateful_functions: Whether the `cond_fn` and `body_fn` functions are stateful functions, i.e., they access class members. None means that if any of the `cond_fn` and `body_fn` is a class method, it will be set to True.

        ## Notice
        1. When using `TracingWhile` and tracing JIT (`core.jit` with `trace=True`), the outer-most `core.jit` must have optional arguments `lazy=False` and `no_cache=False`.
        2. `cond_fn` and `body_fn` must have the same number of arguments.
        3. `cond_fn` and `body_fn` CAN be non-pure functions, i.e., they CAN have side-effects, if `stateful_functions=True`. However, to use non-pure functions, the function inputs shall NOT be class members. See `core.ModuleBase.prepare_control_flow()` for detailed usage of stateful functions.

        ## Warning
        Currently, the in-place modifications to non-local variables of the given non-pure functions CANNOT be JIT traced correctly.
        """
        super().__init__()
        if self.__cache_key__ in _while_object_cache:
            return
        if stateful_functions is None:
            stateful_functions = any(map(lambda fn: hasattr(inspect.unwrap(fn), "__self__"), (body_fn, cond_fn)))
        self.stateful_functions = stateful_functions
        self.cond_fn = cond_fn
        self.body_fn = body_fn
        self._cache_compiled_loop: Dict[
            Tuple[int, torch.dtype, torch.device],
            Tuple[torch.jit.ScriptFunction, UseStateFunc, UseStateFunc],
        ] = {}
        self._cache_compiled_vmap_loop: Dict[
            Tuple[Tuple[int, ...], int, torch.dtype, torch.device],
            Tuple[torch.jit.ScriptFunction, UseStateFunc, UseStateFunc],
        ] = {}
        _while_object_cache[self.__cache_key__] = self
        weakref.finalize(self, _while_object_cache.pop, self.__cache_key__, None)

    @torch.jit.ignore
    def loop(
        self, *x: torch.Tensor | Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, ...] | Tuple[Dict[str, torch.Tensor], Tuple[torch.Tensor, ...]]:
        """
        Executes a while-loop with the given condition and body functions.

        When tracing JIT (`core.jit` with `trace=True`), the `trace_loop` function is used instead; when using `core.vmap`, the `vmap_loop` function is used instead.

        ## Notice
        During normal `torch.jit.script`, this function shall NEVER be invoked for performance-critical paths, please use Python while loop directly.

        :param *x: The input tensors / carry for the loop if `self.stateful_functions=False`; otherwise, firstly a dictionary of tensors containing the state of the `cond_fn` and `body_fn` and then the input tensors / carry for the loop.

        :return: The resulting tensors / carry after the loop completes. If `self.stateful_functions=True`, the resulting tensors / carry are wrapped in a tuple as the second output while the first output is the final state.
        """
        if self.cond_fn is None or self.body_fn is None:
            while self.cond_fn(*x):
                x = self.body_fn(*x)
        else:
            while self.cond_fn(*x):
                x = self.body_fn(*x)
        return x

    @torch.jit.ignore
    def _compile_state_loop_fn(self, original_args: Tuple[torch.Tensor, ...]) -> torch.jit.ScriptFunction:
        state_cond_fn = use_state(lambda: self.cond_fn)
        state_body_fn = use_state(lambda: self.body_fn)
        combined_init_state = state_cond_fn.init_state(False)
        combined_init_state.update(state_body_fn.init_state(False))
        cond_fn = jit(
            state_cond_fn,
            trace=True,
            lazy=False,
            example_inputs=(combined_init_state,) + original_args,
        )
        body_fn = jit(
            state_body_fn,
            trace=True,
            lazy=False,
            example_inputs=(combined_init_state,) + original_args,
        )

        def _loop1(state: Dict[str, torch.Tensor], x1: torch.Tensor):
            cond_state, cond = cond_fn(state, x1)
            state.update(cond_state)
            while cond:
                body_state, x1 = body_fn(state, x1)
                state.update(body_state)
                cond_state, cond = cond_fn(state, x1)
                state.update(cond_state)
            return state, x1

        def _loop2(state: Dict[str, torch.Tensor], x1: torch.Tensor, x2: torch.Tensor):
            xs = (x1, x2)
            cond_state, cond = cond_fn(state, *xs)
            state.update(cond_state)
            while cond:
                body_state, xs = body_fn(state, *xs)
                state.update(body_state)
                cond_state, cond = cond_fn(state, *xs)
                state.update(cond_state)
            return state, xs

        def _loop3(state: Dict[str, torch.Tensor], x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor):
            xs = (x1, x2, x3)
            cond_state, cond = cond_fn(state, *xs)
            state.update(cond_state)
            while cond:
                body_state, xs = body_fn(state, *xs)
                state.update(body_state)
                cond_state, cond = cond_fn(state, xs)
                state.update(cond_state)
            return state, xs

        def _loop4(
            state: Dict[str, torch.Tensor],
            x1: torch.Tensor,
            x2: torch.Tensor,
            x3: torch.Tensor,
            x4: torch.Tensor,
        ):
            xs = (x1, x2, x3, x4)
            cond_state, cond = cond_fn(state, *xs)
            state.update(cond_state)
            while cond:
                body_state, xs = body_fn(state, *xs)
                state.update(body_state)
                cond_state, cond = cond_fn(state, *xs)
                state.update(cond_state)
            return state, xs

        def _loop5(
            state: Dict[str, torch.Tensor],
            x1: torch.Tensor,
            x2: torch.Tensor,
            x3: torch.Tensor,
            x4: torch.Tensor,
            x5: torch.Tensor,
        ):
            xs = (x1, x2, x3, x4, x5)
            cond_state, cond = cond_fn(state, *xs)
            state.update(cond_state)
            while cond:
                body_state, xs = body_fn(state, *xs)
                state.update(body_state)
                cond_state, cond = cond_fn(state, *xs)
                state.update(cond_state)
            return state, xs

        def _loop6(
            state: Dict[str, torch.Tensor],
            x1: torch.Tensor,
            x2: torch.Tensor,
            x3: torch.Tensor,
            x4: torch.Tensor,
            x5: torch.Tensor,
            x6: torch.Tensor,
        ):
            xs = (x1, x2, x3, x4, x5, x6)
            cond_state, cond = cond_fn(state, *xs)
            state.update(cond_state)
            while cond:
                body_state, xs = body_fn(state, *xs)
                state.update(body_state)
                cond_state, cond = cond_fn(state, *xs)
                state.update(cond_state)
            return state, xs

        def _loop7(
            state: Dict[str, torch.Tensor],
            x1: torch.Tensor,
            x2: torch.Tensor,
            x3: torch.Tensor,
            x4: torch.Tensor,
            x5: torch.Tensor,
            x6: torch.Tensor,
            x7: torch.Tensor,
        ):
            xs = (x1, x2, x3, x4, x5, x6, x7)
            cond_state, cond = cond_fn(state, *xs)
            state.update(cond_state)
            while cond:
                body_state, xs = body_fn(state, *xs)
                state.update(body_state)
                cond_state, cond = cond_fn(state, *xs)
                state.update(cond_state)
            return state, xs

        def _loop8(
            state: Dict[str, torch.Tensor],
            x1: torch.Tensor,
            x2: torch.Tensor,
            x3: torch.Tensor,
            x4: torch.Tensor,
            x5: torch.Tensor,
            x6: torch.Tensor,
            x7: torch.Tensor,
            x8: torch.Tensor,
        ):
            xs = (x1, x2, x3, x4, x5, x6, x7, x8)
            cond_state, cond = cond_fn(state, *xs)
            state.update(cond_state)
            while cond:
                body_state, xs = body_fn(state, *xs)
                state.update(body_state)
                cond_state, cond = cond_fn(state, *xs)
                state.update(cond_state)
            return state, xs

        def _loop9(
            state: Dict[str, torch.Tensor],
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
            xs = (x1, x2, x3, x4, x5, x6, x7, x8, x9)
            cond_state, cond = cond_fn(state, *xs)
            state.update(cond_state)
            while cond:
                body_state, xs = body_fn(state, *xs)
                state.update(body_state)
                cond_state, cond = cond_fn(state, *xs)
                state.update(cond_state)
            return state, xs

        loops_dict = {1: _loop1, 2: _loop2, 3: _loop3, 4: _loop4, 5: _loop5, 6: _loop6, 7: _loop7, 8: _loop8, 9: _loop9}

        assert len(original_args) <= len(loops_dict), (
            f"At most {len(loops_dict)} arguments are supported, got {len(original_args)}"
        )
        compiled_loop = torch.jit.script(loops_dict[len(original_args)])
        return compiled_loop, state_cond_fn, state_body_fn

    @torch.jit.ignore
    def _compile_loop_fn(self, original_args: Tuple[torch.Tensor, ...]) -> torch.jit.ScriptFunction:
        cond_fn = jit(self.cond_fn, trace=True, lazy=False, example_inputs=original_args)
        body_fn = jit(self.body_fn, trace=True, lazy=False, example_inputs=original_args)

        def _loop1(x1: torch.Tensor):
            cond = cond_fn(x1)
            while cond:
                x1 = body_fn(x1)
                cond = cond_fn(x1)
            return x1

        def _loop2(x1: torch.Tensor, x2: torch.Tensor):
            xs = (x1, x2)
            cond = cond_fn(*xs)
            while cond:
                xs = body_fn(*xs)
                cond = cond_fn(*xs)
            return xs

        def _loop3(x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor):
            xs = (x1, x2, x3)
            cond = cond_fn(*xs)
            while cond:
                xs = body_fn(*xs)
                cond = cond_fn(*xs)
            return xs

        def _loop4(x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, x4: torch.Tensor):
            xs = (x1, x2, x3, x4)
            cond = cond_fn(*xs)
            while cond:
                xs = body_fn(*xs)
                cond = cond_fn(*xs)
            return xs

        def _loop5(x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, x4: torch.Tensor, x5: torch.Tensor):
            xs = (x1, x2, x3, x4, x5)
            cond = cond_fn(*xs)
            while cond:
                xs = body_fn(*xs)
                cond = cond_fn(*xs)
            return xs

        def _loop6(x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, x4: torch.Tensor, x5: torch.Tensor, x6: torch.Tensor):
            xs = (x1, x2, x3, x4, x5, x6)
            cond = cond_fn(*xs)
            while cond:
                xs = body_fn(*xs)
                cond = cond_fn(*xs)
            return xs

        def _loop7(
            x1: torch.Tensor,
            x2: torch.Tensor,
            x3: torch.Tensor,
            x4: torch.Tensor,
            x5: torch.Tensor,
            x6: torch.Tensor,
            x7: torch.Tensor,
        ):
            xs = (x1, x2, x3, x4, x5, x6, x7)
            cond = cond_fn(*xs)
            while cond:
                xs = body_fn(*xs)
                cond = cond_fn(*xs)
            return xs

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
            xs = (x1, x2, x3, x4, x5, x6, x7, x8)
            cond = cond_fn(*xs)
            while cond:
                xs = body_fn(*xs)
                cond = cond_fn(*xs)
            return xs

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
            xs = (x1, x2, x3, x4, x5, x6, x7, x8, x9)
            cond = cond_fn(*xs)
            while cond:
                xs = body_fn(*xs)
                cond = cond_fn(*xs)
            return xs

        loops_dict = {1: _loop1, 2: _loop2, 3: _loop3, 4: _loop4, 5: _loop5, 6: _loop6, 7: _loop7, 8: _loop8, 9: _loop9}

        assert len(original_args) <= len(loops_dict), (
            f"At most {len(loops_dict)} arguments are supported, got {len(original_args)}"
        )
        compiled_loop = torch.jit.script(loops_dict[len(original_args)])
        return compiled_loop

    @trace_impl(loop)
    def trace_loop(self, *x: torch.Tensor | Dict[str, torch.Tensor]):
        if self.stateful_functions:
            state: Dict[str, torch.Tensor] = x[0]
            x = x[1:]
        else:
            state = None
        key = tuple((a.ndim, a.dtype, a.device) for a in x)
        if state is not None:
            if key in self._cache_compiled_loop:
                compiled_cond = self._cache_compiled_loop[key]
            else:
                compiled_cond, _, _ = self._compile_state_loop_fn(x)
                self._cache_compiled_loop[key] = compiled_cond
            res = compiled_cond(state, *x)
        else:
            if key in self._cache_compiled_loop:
                compiled_cond = self._cache_compiled_loop[key]
            else:
                compiled_cond = self._compile_loop_fn(x)
                self._cache_compiled_loop[key] = compiled_cond
            res = compiled_cond(*x)
        return res

    @torch.jit.ignore
    def _compile_vmap_loop_fn(self, originals: Tuple[torch.Tensor, ...], vmap_dims: Tuple[Tuple[int, ...], ...]):
        cond_dims = ()
        new_vmap_dims = ()

        # vmap
        def _wrap_inputs(args):
            return tuple(_vmap_fix.wrap_batch_tensor(a, d) for a, d in zip(args, vmap_dims))

        def _vmap_cond(*args: torch.Tensor) -> Tuple[torch.Tensor, ...]:
            cond_res: torch.Tensor = self.cond_fn(*_wrap_inputs(args))
            assert cond_res.ndim == 0, f"Expected scalar condition, got ndim={cond_res.ndim}"
            nonlocal cond_dims
            cond_res, cond_dims, _ = _vmap_fix.unwrap_batch_tensor(cond_res)
            return cond_res

        def _vmap_body(*args):
            body_res = self.body_fn(*_wrap_inputs(args))
            body_out = []
            body_dims = []
            for b in body_res:
                b, d, _ = _vmap_fix.unwrap_batch_tensor(b)
                body_out.append(b)
                body_dims.append(d)
            nonlocal new_vmap_dims
            new_vmap_dims = tuple(body_dims)
            return tuple(body_out)

        def _vmap_select(cond: torch.Tensor, new: Tuple[torch.Tensor, ...], old: Tuple[torch.Tensor, ...]) -> torch.Tensor:
            batched_cond = _vmap_fix.wrap_batch_tensor(cond, cond_dims)
            returns = []
            for a, b, a_dim, b_dim in zip(new, old, new_vmap_dims, vmap_dims):
                a = _vmap_fix.wrap_batch_tensor(a, a_dim)
                b = _vmap_fix.wrap_batch_tensor(b, b_dim)
                if _vmap_fix.is_batched_tensor(a) or _vmap_fix.is_batched_tensor(b):
                    returns.append(torch.where(batched_cond, a, b))
                else:
                    returns.append(torch.where(cond.any(), a, b))
                returns[-1] = _vmap_fix.unwrap_batch_tensor(returns[-1])[0]
            return tuple(returns)

        # JIT
        init_cond, dummy_cond_out = jit(_vmap_cond, trace=True, example_inputs=originals, return_dummy_output=True)
        init_body, dummy_body_out = jit(_vmap_body, trace=True, example_inputs=originals, return_dummy_output=True)
        init_select, originals = jit(
            _vmap_select, trace=True, example_inputs=(dummy_cond_out, dummy_body_out, originals), return_dummy_output=True
        )
        vmap_dims = new_vmap_dims
        cond = jit(_vmap_cond, trace=True, example_inputs=originals)
        body = jit(_vmap_body, trace=True, example_inputs=originals)
        select = jit(_vmap_select, trace=True, example_inputs=(dummy_cond_out, dummy_body_out, originals))

        def _loop1(x1: torch.Tensor):
            cond_res = init_cond(x1)
            if not cond_res.any():
                return x1
            x1_new = init_body(x1)
            x1 = init_select(cond_res, (x1_new,), (x1,))[0]
            cond_res = cond(x1)
            while cond_res.any():
                x1_new = body(x1)
                x1 = select(cond_res, (x1_new,), (x1,))[0]
                cond_res = cond(x1)
            return x1

        def _loop2(x1: torch.Tensor, x2: torch.Tensor):
            cond_res = init_cond(x1, x2)
            if not cond_res.any():
                return x1, x2
            x1_new, x2_new = init_body(x1, x2)
            x1, x2 = init_select(cond_res, (x1_new, x2_new), (x1, x2))
            cond_res = cond(x1, x2)
            while cond_res.any():
                x1_new, x2_new = body(x1, x2)
                x1, x2 = select(cond_res, (x1_new, x2_new), (x1, x2))
                cond_res = cond(x1, x2)
            return x1, x2

        def _loop3(x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor):
            cond_res = init_cond(x1, x2, x3)
            if not cond_res.any():
                return x1, x2, x3
            x1_new, x2_new, x3_new = init_body(x1, x2, x3)
            x1, x2, x3 = init_select(cond_res, (x1_new, x2_new, x3_new), (x1, x2, x3))
            cond_res = cond(x1, x2, x3)
            while cond_res.any():
                x1_new, x2_new, x3_new = body(x1, x2, x3)
                x1, x2, x3 = select(cond_res, (x1_new, x2_new, x3_new), (x1, x2, x3))
                cond_res = cond(x1, x2, x3)
            return x1, x2, x3

        def _loop4(x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, x4: torch.Tensor):
            cond_res = init_cond(x1, x2, x3, x4)
            if not cond_res.any():
                return x1, x2, x3, x4
            x1_new, x2_new, x3_new, x4_new = init_body(x1, x2, x3, x4)
            x1, x2, x3, x4 = init_select(cond_res, (x1_new, x2_new, x3_new, x4_new), (x1, x2, x3, x4))
            cond_res = cond(x1, x2, x3, x4)
            while cond_res.any():
                x1_new, x2_new, x3_new, x4_new = body(x1, x2, x3, x4)
                x1, x2, x3, x4 = select(cond_res, (x1_new, x2_new, x3_new, x4_new), (x1, x2, x3, x4))
                cond_res = cond(x1, x2, x3, x4)
            return x1, x2, x3, x4

        def _loop5(x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, x4: torch.Tensor, x5: torch.Tensor):
            cond_res = init_cond(x1, x2, x3, x4, x5)
            if not cond_res.any():
                return x1, x2, x3, x4, x5
            x1_new, x2_new, x3_new, x4_new, x5_new = init_body(x1, x2, x3, x4, x5)
            x1, x2, x3, x4, x5 = init_select(cond_res, (x1_new, x2_new, x3_new, x4_new, x5_new), (x1, x2, x3, x4, x5))
            cond_res = cond(x1, x2, x3, x4, x5)
            while cond_res.any():
                x1_new, x2_new, x3_new, x4_new, x5_new = body(x1, x2, x3, x4, x5)
                x1, x2, x3, x4, x5 = select(cond_res, (x1_new, x2_new, x3_new, x4_new, x5_new), (x1, x2, x3, x4, x5))
                cond_res = cond(x1, x2, x3, x4, x5)
            return x1, x2, x3, x4, x5

        def _loop6(x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, x4: torch.Tensor, x5: torch.Tensor, x6: torch.Tensor):
            cond_res = init_cond(x1, x2, x3, x4, x5, x6)
            if not cond_res.any():
                return x1, x2, x3, x4, x5, x6
            x1_new, x2_new, x3_new, x4_new, x5_new, x6_new = init_body(x1, x2, x3, x4, x5, x6)
            x1, x2, x3, x4, x5, x6 = init_select(
                cond_res, (x1_new, x2_new, x3_new, x4_new, x5_new, x6_new), (x1, x2, x3, x4, x5, x6)
            )
            cond_res = cond(x1, x2, x3, x4, x5, x6)
            while cond_res.any():
                x1_new, x2_new, x3_new, x4_new, x5_new, x6_new = body(x1, x2, x3, x4, x5, x6)
                x1, x2, x3, x4, x5, x6 = select(
                    cond_res, (x1_new, x2_new, x3_new, x4_new, x5_new, x6_new), (x1, x2, x3, x4, x5, x6)
                )
                cond_res = cond(x1, x2, x3, x4, x5, x6)
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
            cond_res = init_cond(x1, x2, x3, x4, x5, x6, x7)
            if not cond_res.any():
                return x1, x2, x3, x4, x5, x6, x7
            x1_new, x2_new, x3_new, x4_new, x5_new, x6_new, x7_new = init_body(x1, x2, x3, x4, x5, x6, x7)
            x1, x2, x3, x4, x5, x6, x7 = init_select(
                cond_res, (x1_new, x2_new, x3_new, x4_new, x5_new, x6_new, x7_new), (x1, x2, x3, x4, x5, x6, x7)
            )
            cond_res = cond(x1, x2, x3, x4, x5, x6, x7)
            while cond_res.any():
                x1_new, x2_new, x3_new, x4_new, x5_new, x6_new, x7_new = body(x1, x2, x3, x4, x5, x6, x7)
                x1, x2, x3, x4, x5, x6, x7 = select(
                    cond_res, (x1_new, x2_new, x3_new, x4_new, x5_new, x6_new, x7_new), (x1, x2, x3, x4, x5, x6, x7)
                )
                cond_res = cond(x1, x2, x3, x4, x5, x6, x7)
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
            cond_res = init_cond(x1, x2, x3, x4, x5, x6, x7, x8)
            if not cond_res.any():
                return x1, x2, x3, x4, x5, x6, x7, x8
            x1_new, x2_new, x3_new, x4_new, x5_new, x6_new, x7_new, x8_new = init_body(x1, x2, x3, x4, x5, x6, x7, x8)
            x1, x2, x3, x4, x5, x6, x7, x8 = init_select(
                cond_res, (x1_new, x2_new, x3_new, x4_new, x5_new, x6_new, x7_new, x8_new), (x1, x2, x3, x4, x5, x6, x7, x8)
            )
            cond_res = cond(x1, x2, x3, x4, x5, x6, x7, x8)
            while cond_res.any():
                x1_new, x2_new, x3_new, x4_new, x5_new, x6_new, x7_new, x8_new = body(x1, x2, x3, x4, x5, x6, x7, x8)
                x1, x2, x3, x4, x5, x6, x7, x8 = select(
                    cond_res, (x1_new, x2_new, x3_new, x4_new, x5_new, x6_new, x7_new, x8_new), (x1, x2, x3, x4, x5, x6, x7, x8)
                )
                cond_res = cond(x1, x2, x3, x4, x5, x6, x7, x8)
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
            cond_res = init_cond(x1, x2, x3, x4, x5, x6, x7, x8, x9)
            if not cond_res.any():
                return x1, x2, x3, x4, x5, x6, x7, x8, x9
            x1_new, x2_new, x3_new, x4_new, x5_new, x6_new, x7_new, x8_new, x9_new = init_body(
                x1, x2, x3, x4, x5, x6, x7, x8, x9
            )
            x1, x2, x3, x4, x5, x6, x7, x8, x9 = init_select(
                cond_res,
                (x1_new, x2_new, x3_new, x4_new, x5_new, x6_new, x7_new, x8_new, x9_new),
                (x1, x2, x3, x4, x5, x6, x7, x8, x9),
            )
            cond_res = cond(x1, x2, x3, x4, x5, x6, x7, x8, x9)
            while cond_res.any():
                x1_new, x2_new, x3_new, x4_new, x5_new, x6_new, x7_new, x8_new, x9_new = body(
                    x1, x2, x3, x4, x5, x6, x7, x8, x9
                )
                x1, x2, x3, x4, x5, x6, x7, x8, x9 = select(
                    cond_res,
                    (x1_new, x2_new, x3_new, x4_new, x5_new, x6_new, x7_new, x8_new, x9_new),
                    (x1, x2, x3, x4, x5, x6, x7, x8, x9),
                )
                cond_res = cond(x1, x2, x3, x4, x5, x6, x7, x8, x9)
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
        assert len(originals) <= len(loops_dict), f"At most {len(loops_dict)} arguments are supported, got {len(originals)}"
        return torch.jit.script(loops_dict[len(originals)]), vmap_dims

    @vmap_impl(loop)
    def vmap_loop(self, *x: torch.Tensor) -> torch.Tensor:
        # get vmap dims and original arguments
        vmap_dims = []
        original_args: List[torch.Tensor] = []
        for arg in x:
            assert isinstance(arg, torch.Tensor), f"Expect all arguments in `vmap` to be `torch.Tensor`, got {type(arg)}"
            arg, in_dim, _ = _vmap_fix.unwrap_batch_tensor(arg)
            vmap_dims.append(in_dim)
            original_args.append(arg)
        original_args = tuple(original_args)
        # compile
        key = tuple((d, a.ndim, a.dtype, a.device) for d, a in zip(vmap_dims, original_args))
        if key in self._cache_compiled_vmap_loop:
            vmap_loop_compiled, final_vmap_dims = self._cache_compiled_vmap_loop[key]
        else:
            vmap_loop_compiled, final_vmap_dims = self._compile_vmap_loop_fn(original_args, vmap_dims)
            self._cache_compiled_vmap_loop[key] = (vmap_loop_compiled, final_vmap_dims)
        ret = vmap_loop_compiled(*original_args)
        returns = []
        for r, d in zip(ret, final_vmap_dims):
            returns.append(_vmap_fix.wrap_batch_tensor(r, d))
        return tuple(returns) if len(returns) > 1 else returns[0]
