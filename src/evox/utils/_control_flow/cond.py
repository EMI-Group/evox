import inspect
import weakref
from typing import Dict, List, Sequence, Tuple

import torch

from ...core import ModuleBase, jit, jit_class, trace_impl, use_state, vmap_impl
from .utils import VarArgsCallableMultiRet, _get_cache_key_object, _param_clone

_cond_object_cache = weakref.WeakValueDictionary()


@jit_class
class TracingCond(ModuleBase):
    """A helper class used to trace an if-else control flow."""

    def __new__(cls, true_fn, false_fn, stateful_functions: bool | None = None):
        key_or_obj = _get_cache_key_object(_cond_object_cache, true_fn, false_fn)
        if isinstance(key_or_obj, tuple):
            obj = super().__new__(cls)
            obj.__cache_key__ = key_or_obj
            return obj
        else:
            return key_or_obj

    def __init__(
        self, true_fn: VarArgsCallableMultiRet, false_fn: VarArgsCallableMultiRet, stateful_functions: bool | None = None
    ):
        """
        Initialize the `TracingCond`.

        :param true_fn: The true branch function.
        :param false_fn: The false branch function.
        :param stateful_functions: Whether the `true_fn` and `false_fn` functions are stateful functions, i.e., they access class members. None means that if any of the `true_fn` and `false_fn` is a class method, it will be set to True.

        ## Notice
        1. When using `TracingCond` and tracing JIT (`core.jit` with `trace=True`), the outer-most `core.jit` must have optional arguments `lazy=False` and `no_cache=False`.
        2. `true_fn` and `false_fn` must have the same number of arguments.
        3. `true_fn` and `false_fn` CAN be non-pure functions, i.e., they CAN have side-effects, if `stateful_functions=True`. However, to use non-pure functions, the function inputs shall NOT be class members. See `core.ModuleBase.prepare_control_flow()` for detailed usage of stateful functions.

        ## Warning
        Currently, the in-place modifications to non-local variables of the given non-pure functions CANNOT be JIT traced correctly.
        """
        super().__init__()
        if self.__cache_key__ in _cond_object_cache:
            return
        if stateful_functions is None:
            stateful_functions = any(map(lambda fn: hasattr(inspect.unwrap(fn), "__self__"), (true_fn, false_fn)))
        self.stateful_functions = stateful_functions
        self.true_fn = true_fn
        self.false_fn = false_fn
        self._cache_compiled_cond: Dict[Tuple[int, torch.dtype, torch.device], torch.jit.ScriptFunction] = {}
        _cond_object_cache[self.__cache_key__] = self
        weakref.finalize(self, _cond_object_cache.pop, self.__cache_key__, None)

    @torch.jit.ignore
    def cond(
        self, cond: torch.Tensor | Dict[str, torch.Tensor], *x: torch.Tensor
    ) -> (
        Tuple[Dict[str, torch.Tensor], List[torch.Tensor] | torch.Tensor]
        | Dict[str, torch.Tensor]
        | List[torch.Tensor]
        | torch.Tensor
        | None
    ):
        """Runs either the true or false branch based on the given condition.

        When tracing JIT (`core.jit` with `trace=True`), the `trace_cond` function is used instead; when using `core.vmap`, the `vmap_cond` function is used instead.

        ## Notice
        During normal `torch.jit.script`, this function shall NEVER be invoked for performance-critical paths, please use Python if-else directly.

        :param cond: An bool tensor that indicates which branch to run if `self.stateful_functions=False`; otherwise, a dictionary of tensors containing the state of the branch functions.
        :param *x: The input tensors to the branch functions if `self.stateful_functions=False`; otherwise, firstly the `branch_idx` tensor and then the input tensors to the branch functions.

        :return: The output tensors from the chosen branch function. If `self.stateful_functions=True`, the output tensors are wrapped in a tuple as the second output while the first output is the state of the executed branch function.
        """
        if cond:
            x = self.true_fn(*x)
        else:
            x = self.false_fn(*x)
        return x

    @torch.jit.ignore
    def _compile_cond_fn(self, original_args: Tuple[torch.Tensor, ...]):
        true_fn = jit(
            self.true_fn,
            trace=True,
            lazy=False,
            example_inputs=original_args,
        )
        false_fn = jit(
            self.false_fn,
            trace=True,
            lazy=False,
            example_inputs=original_args,
        )

        def _cond1(condition: torch.Tensor, x1: torch.Tensor):
            if condition:
                x = true_fn(x1)
            else:
                x = false_fn(x1)
            return x

        def _cond2(
            condition: torch.Tensor,
            x1: torch.Tensor,
            x2: torch.Tensor,
        ):
            if condition:
                x = true_fn(x1, x2)
            else:
                x = false_fn(x1, x2)
            return x

        def _cond3(condition: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor):
            if condition:
                x = true_fn(x1, x2, x3)
            else:
                x = false_fn(x1, x2, x3)
            return x

        def _cond4(condition: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, x4: torch.Tensor):
            if condition:
                x = true_fn(x1, x2, x3, x4)
            else:
                x = false_fn(x1, x2, x3, x4)
            return x

        def _cond5(
            condition: torch.Tensor,
            x1: torch.Tensor,
            x2: torch.Tensor,
            x3: torch.Tensor,
            x4: torch.Tensor,
            x5: torch.Tensor,
        ):
            if condition:
                x = true_fn(x1, x2, x3, x4, x5)
            else:
                x = false_fn(x1, x2, x3, x4, x5)
            return x

        def _cond6(
            condition: torch.Tensor,
            x1: torch.Tensor,
            x2: torch.Tensor,
            x3: torch.Tensor,
            x4: torch.Tensor,
            x5: torch.Tensor,
            x6: torch.Tensor,
        ):
            if condition:
                x = true_fn(x1, x2, x3, x4, x5, x6)
            else:
                x = false_fn(x1, x2, x3, x4, x5, x6)
            return x

        def _cond7(
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
                x = true_fn(x1, x2, x3, x4, x5, x6, x7)
            else:
                x = false_fn(x1, x2, x3, x4, x5, x6, x7)
            return x

        def _cond8(
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
                x = true_fn(x1, x2, x3, x4, x5, x6, x7, x8)
            else:
                x = false_fn(x1, x2, x3, x4, x5, x6, x7, x8)
            return x

        def _cond9(
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
                x = true_fn(x1, x2, x3, x4, x5, x6, x7, x8, x9)
            else:
                x = false_fn(x1, x2, x3, x4, x5, x6, x7, x8, x9)
            return x

        cond_dict = {1: _cond1, 2: _cond2, 3: _cond3, 4: _cond4, 5: _cond5, 6: _cond6, 7: _cond7, 8: _cond8, 9: _cond9}
        assert len(original_args) <= len(
            cond_dict
        ), f"At most {len(cond_dict)} arguments are supported, got {len(original_args)}"
        compiled_cond = torch.jit.script(cond_dict[len(original_args)])
        return compiled_cond

    @torch.jit.ignore
    def _compile_state_cond_fn(self, original_args: Tuple[torch.Tensor, ...]):
        # use state for in-place modifications
        state_true_fn = use_state(lambda: self.true_fn)
        state_false_fn = use_state(lambda: self.false_fn)
        true_fn = jit(
            state_true_fn,
            trace=True,
            lazy=False,
            example_inputs=(state_true_fn.init_state(False),) + original_args,
        )
        false_fn = jit(
            state_false_fn,
            trace=True,
            lazy=False,
            example_inputs=(state_false_fn.init_state(False),) + original_args,
        )

        def _cond1(
            state: Dict[str, torch.Tensor],
            condition: torch.Tensor,
            x1: torch.Tensor,
        ):
            if condition:
                x = true_fn(state, x1)
            else:
                x = false_fn(state, x1)
            return x

        def _cond2(
            state: Dict[str, torch.Tensor],
            condition: torch.Tensor,
            x1: torch.Tensor,
            x2: torch.Tensor,
        ):
            if condition:
                x = true_fn(state, x1, x2)
            else:
                x = false_fn(state, x1, x2)
            return x

        def _cond3(
            state: Dict[str, torch.Tensor],
            condition: torch.Tensor,
            x1: torch.Tensor,
            x2: torch.Tensor,
            x3: torch.Tensor,
        ):
            if condition:
                x = true_fn(state, x1, x2, x3)
            else:
                x = false_fn(state, x1, x2, x3)
            return x

        def _cond4(
            state: Dict[str, torch.Tensor],
            condition: torch.Tensor,
            x1: torch.Tensor,
            x2: torch.Tensor,
            x3: torch.Tensor,
            x4: torch.Tensor,
        ):
            if condition:
                x = true_fn(state, x1, x2, x3, x4)
            else:
                x = false_fn(state, x1, x2, x3, x4)
            return x

        def _cond5(
            state: Dict[str, torch.Tensor],
            condition: torch.Tensor,
            x1: torch.Tensor,
            x2: torch.Tensor,
            x3: torch.Tensor,
            x4: torch.Tensor,
            x5: torch.Tensor,
        ):
            if condition:
                x = true_fn(state, x1, x2, x3, x4, x5)
            else:
                x = false_fn(state, x1, x2, x3, x4, x5)
            return x

        def _cond6(
            state: Dict[str, torch.Tensor],
            condition: torch.Tensor,
            x1: torch.Tensor,
            x2: torch.Tensor,
            x3: torch.Tensor,
            x4: torch.Tensor,
            x5: torch.Tensor,
            x6: torch.Tensor,
        ):
            if condition:
                x = true_fn(state, x1, x2, x3, x4, x5, x6)
            else:
                x = false_fn(state, x1, x2, x3, x4, x5, x6)
            return x

        def _cond7(
            state: Dict[str, torch.Tensor],
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
                x = true_fn(state, x1, x2, x3, x4, x5, x6, x7)
            else:
                x = false_fn(state, x1, x2, x3, x4, x5, x6, x7)
            return x

        def _cond8(
            state: Dict[str, torch.Tensor],
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
                x = true_fn(state, x1, x2, x3, x4, x5, x6, x7, x8)
            else:
                x = false_fn(state, x1, x2, x3, x4, x5, x6, x7, x8)
            return x

        def _cond9(
            state: Dict[str, torch.Tensor],
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
                x = true_fn(state, x1, x2, x3, x4, x5, x6, x7, x8, x9)
            else:
                x = false_fn(state, x1, x2, x3, x4, x5, x6, x7, x8, x9)
            return x

        cond_dict = {1: _cond1, 2: _cond2, 3: _cond3, 4: _cond4, 5: _cond5, 6: _cond6, 7: _cond7, 8: _cond8, 9: _cond9}
        assert len(original_args) <= len(
            cond_dict
        ), f"At most {len(cond_dict)} arguments are supported, got {len(original_args)}"
        compiled_cond = torch.jit.script(cond_dict[len(original_args)])
        return compiled_cond, state_true_fn, state_false_fn

    @trace_impl(cond)
    def trace_cond(self, cond: torch.Tensor | Dict[str, torch.Tensor], *x: torch.Tensor):
        if self.stateful_functions:
            state: Dict[str, torch.Tensor] = cond
            cond = x[0]
            x = x[1:]
        else:
            state = None
        key = tuple((a.ndim, a.dtype, a.device) for a in (cond,) + x)
        if state is not None:
            if key in self._cache_compiled_cond:
                compiled_cond = self._cache_compiled_cond[key]
            else:
                compiled_cond, _, _ = self._compile_state_cond_fn(x)
                self._cache_compiled_cond[key] = compiled_cond
            res = compiled_cond(state, cond, *x)
        else:
            if key in self._cache_compiled_cond:
                compiled_cond = self._cache_compiled_cond[key]
            else:
                compiled_cond = self._compile_cond_fn(x)
                self._cache_compiled_cond[key] = compiled_cond
            res = compiled_cond(cond, *x)
        return res

    @vmap_impl(cond)
    def vmap_cond(self, cond: torch.Tensor | Dict[str, torch.Tensor], *x: torch.Tensor):
        # cannot dynamic dispatch for vmap, change to torch.where
        # use_state to support in-place modifications
        if self.stateful_functions:
            state: Dict[str, torch.Tensor] = cond
            cond = x[0]
            x = x[1:]
            ori_state = {k: _param_clone(v) for k, v in state.items()}
        else:
            state = None
        branch_results: List[torch.Tensor | Tuple[torch.Tensor, ...] | None] = []
        if state is not None:
            final_states: List[Dict[str, torch.Tensor]] = []
        for branch_fn in (self.true_fn, self.false_fn):
            if state is None:
                result = branch_fn(*x)
            else:
                fn = use_state(lambda: branch_fn)
                branch_cloned_state = {k: _param_clone(state[k]) for k in fn.init_state(False)}
                result = fn(branch_cloned_state, *x)
                if isinstance(result, tuple):
                    new_state, result = result
                else:
                    new_state = result
                    result = None
            if len(branch_results) > 0:
                assert branch_results[-1] is result or (
                    type(branch_results[-1]) is type(result)
                    and not isinstance(result, tuple)
                    or len(result) == len(branch_results[-1])
                ), "Branch functions should return the same type of outputs."
            branch_results.append(result)
            if state is not None:
                final_states.append(new_state)
        # get conditional outputs
        if branch_results[0] is None:
            final_output = None
        elif isinstance(branch_results[0], torch.Tensor):
            final_output = torch.where(cond, branch_results[0], branch_results[1])
        elif isinstance(branch_results[0], Sequence):
            final_output = []
            for results in zip(*branch_results):
                final_output.append(torch.where(cond, results[0], results[1]))
        else:
            raise ValueError("The type of returns of branches should be the same.")
        if state is None:
            return final_output
        else:
            # set conditional state
            state_out: Dict[str, torch.Tensor] = {}
            for k, v in ori_state.items():
                final_state_tensors = [s.get(k, v) for s in final_states]
                state_out[k] = torch.where(cond, final_state_tensors[0], final_state_tensors[1])
            # return
            return state_out, final_output
