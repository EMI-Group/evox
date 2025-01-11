import inspect
import weakref
from typing import Dict, List, Sequence, Tuple

import torch

from ...core import ModuleBase, jit, jit_class, trace_impl, use_state, vmap_impl
from ...core.module import UseStateFunc
from ..jit_fix_operator import switch as _switch
from .utils import VarArgsCallableMultiRet, _get_cache_key_object, _param_clone

_switch_object_cache = weakref.WeakValueDictionary()


@jit_class
class TracingSwitch(ModuleBase):
    """A helper class used to trace an match-case (switch-case) control flow."""

    def __new__(cls, *branch_fns, stateful_functions: bool | None = None):
        key_or_obj = _get_cache_key_object(_switch_object_cache, *branch_fns)
        if isinstance(key_or_obj, tuple):
            obj = super().__new__(cls)
            obj.__cache_key__ = key_or_obj
            return obj
        else:
            return key_or_obj

    def __init__(self, *branch_fns: VarArgsCallableMultiRet, stateful_functions: bool | None = None):
        """
        Initialize the `TracingCond`.

        :param branch_fns: The branch functions.
        :param stateful_functions: Whether the `branch_fns` functions are stateful functions, i.e., they access class members. None means that if any of the `branch_fns` is a class method, it will be set to True.

        ## Notice
        1. When using `TracingCond` and tracing JIT (`core.jit` with `trace=True`), the outer-most `core.jit` must have optional arguments `lazy=False` and `no_cache=False`.
        2. `branch_fns` must have the same number of arguments.
        3. `branch_fns` CAN be non-pure functions, i.e., they CAN have side-effects, if `stateful_functions=True`. However, to use non-pure functions, the function inputs shall NOT be class members. See `core.ModuleBase.prepare_control_flow()` for detailed usage of stateful functions.

        ## Warning
        Currently, the in-place modifications to non-local variables of the given non-pure functions CANNOT be JIT traced correctly.
        """
        super().__init__()
        if self.__cache_key__ in _switch_object_cache:
            return
        assert len(branch_fns) >= 2, f"At least 2 branches are required, got {len(branch_fns)}"
        if stateful_functions is None:
            stateful_functions = any(map(lambda fn: hasattr(inspect.unwrap(fn), "__self__"), branch_fns))
        self.stateful_functions = stateful_functions
        self.branch_fns = branch_fns
        self._cache_compiled_switch: Dict[Tuple[int, torch.dtype, torch.device], torch.jit.ScriptFunction] = {}
        _switch_object_cache[self.__cache_key__] = self
        weakref.finalize(self, _switch_object_cache.pop, self.__cache_key__, None)

    @torch.jit.ignore
    def switch(
        self, branch_idx: torch.Tensor | Dict[str, torch.Tensor], *x: torch.Tensor
    ) -> (
        Tuple[Dict[str, torch.Tensor], List[torch.Tensor] | torch.Tensor]
        | Dict[str, torch.Tensor]
        | List[torch.Tensor]
        | torch.Tensor
        | None
    ):
        """Runs the selected branch based on the given branch index.

        When tracing JIT (`core.jit` with `trace=True`), the `trace_switch` function is used instead; when using `core.vmap`, the `vmap_switch` function is used instead.

        ## Notice
        During normal `torch.jit.script`, this function shall NEVER be invoked for performance-critical paths, please use Python if-else directly.

        :param branch_idx: An int tensor that indicates which branch to run if `self.stateful_functions=False`; otherwise, a dictionary of tensors containing the state of the branch functions.
        :param *x: The input tensors to the branch functions if `self.stateful_functions=False`; otherwise, firstly the `branch_idx` tensor and then the input tensors to the branch functions.

        :return: The output tensors from the chosen branch function. If `self.stateful_functions=True`, the output tensors are wrapped in a tuple as the second output while the first output is the state of the executed branch function.
        """
        return self.branch_fns[branch_idx.item()](*x)

    @torch.jit.ignore
    def _compile_switch_fn(self, original_args: Tuple[torch.Tensor, ...]):
        assert len(self.branch_fns) <= 9, f"At most 9 branches are supported, got {len(self.branch_fns)}"
        branch_fns: List[torch.jit.ScriptFunction] = []
        for branch_fn in self.branch_fns:
            branch_fns.append(jit(branch_fn, trace=True, lazy=False, example_inputs=original_args))
        # set local functions
        branch_fn0, branch_fn1, *_rest = branch_fns
        if len(_rest) > 0:
            branch_fn2, *_rest = _rest
        else:
            branch_fn2 = branch_fn0
        if len(_rest) > 0:
            branch_fn3, *_rest = _rest
        else:
            branch_fn3 = branch_fn0
        if len(_rest) > 0:
            branch_fn4, *_rest = _rest
        else:
            branch_fn4 = branch_fn0
        if len(_rest) > 0:
            branch_fn5, *_rest = _rest
        else:
            branch_fn5 = branch_fn0
        if len(_rest) > 0:
            branch_fn6, *_rest = _rest
        else:
            branch_fn6 = branch_fn0
        if len(_rest) > 0:
            branch_fn7, *_rest = _rest
        else:
            branch_fn7 = branch_fn0
        if len(_rest) > 0:
            branch_fn8, *_rest = _rest
        else:
            branch_fn8 = branch_fn0

        def _switch1(branch_idx: torch.Tensor, x1: torch.Tensor):
            if branch_idx == 0:
                return branch_fn0(x1)
            if branch_idx == 1:
                return branch_fn1(x1)
            if branch_idx == 2:
                return branch_fn2(x1)
            if branch_idx == 3:
                return branch_fn3(x1)
            if branch_idx == 4:
                return branch_fn4(x1)
            if branch_idx == 5:
                return branch_fn5(x1)
            if branch_idx == 6:
                return branch_fn6(x1)
            if branch_idx == 7:
                return branch_fn7(x1)
            if branch_idx == 8:
                return branch_fn8(x1)
            raise AssertionError(f"Invalid branch index {branch_idx.item()}")

        def _switch2(branch_idx: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor):
            if branch_idx == 0:
                return branch_fn0(x1, x2)
            if branch_idx == 1:
                return branch_fn1(x1, x2)
            if branch_idx == 2:
                return branch_fn2(x1, x2)
            if branch_idx == 3:
                return branch_fn3(x1, x2)
            if branch_idx == 4:
                return branch_fn4(x1, x2)
            if branch_idx == 5:
                return branch_fn5(x1, x2)
            if branch_idx == 6:
                return branch_fn6(x1, x2)
            if branch_idx == 7:
                return branch_fn7(x1, x2)
            if branch_idx == 8:
                return branch_fn8(x1, x2)
            raise AssertionError(f"Invalid branch index {branch_idx.item()}")

        def _switch3(branch_idx: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor):
            if branch_idx == 0:
                return branch_fn0(x1, x2, x3)
            if branch_idx == 1:
                return branch_fn1(x1, x2, x3)
            if branch_idx == 2:
                return branch_fn2(x1, x2, x3)
            if branch_idx == 3:
                return branch_fn3(x1, x2, x3)
            if branch_idx == 4:
                return branch_fn4(x1, x2, x3)
            if branch_idx == 5:
                return branch_fn5(x1, x2, x3)
            if branch_idx == 6:
                return branch_fn6(x1, x2, x3)
            if branch_idx == 7:
                return branch_fn7(x1, x2, x3)
            if branch_idx == 8:
                return branch_fn8(x1, x2, x3)
            raise AssertionError(f"Invalid branch index {branch_idx.item()}")

        def _switch4(branch_idx: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, x4: torch.Tensor):
            if branch_idx == 0:
                return branch_fn0(x1, x2, x3, x4)
            if branch_idx == 1:
                return branch_fn1(x1, x2, x3, x4)
            if branch_idx == 2:
                return branch_fn2(x1, x2, x3, x4)
            if branch_idx == 3:
                return branch_fn3(x1, x2, x3, x4)
            if branch_idx == 4:
                return branch_fn4(x1, x2, x3, x4)
            if branch_idx == 5:
                return branch_fn5(x1, x2, x3, x4)
            if branch_idx == 6:
                return branch_fn6(x1, x2, x3, x4)
            if branch_idx == 7:
                return branch_fn7(x1, x2, x3, x4)
            if branch_idx == 8:
                return branch_fn8(x1, x2, x3, x4)
            raise AssertionError(f"Invalid branch index {branch_idx.item()}")

        def _switch5(
            branch_idx: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, x4: torch.Tensor, x5: torch.Tensor
        ):
            if branch_idx == 0:
                return branch_fn0(x1, x2, x3, x4, x5)
            if branch_idx == 1:
                return branch_fn1(x1, x2, x3, x4, x5)
            if branch_idx == 2:
                return branch_fn2(x1, x2, x3, x4, x5)
            if branch_idx == 3:
                return branch_fn3(x1, x2, x3, x4, x5)
            if branch_idx == 4:
                return branch_fn4(x1, x2, x3, x4, x5)
            if branch_idx == 5:
                return branch_fn5(x1, x2, x3, x4, x5)
            if branch_idx == 6:
                return branch_fn6(x1, x2, x3, x4, x5)
            if branch_idx == 7:
                return branch_fn7(x1, x2, x3, x4, x5)
            if branch_idx == 8:
                return branch_fn8(x1, x2, x3, x4, x5)
            raise AssertionError(f"Invalid branch index {branch_idx.item()}")

        def _switch6(
            branch_idx: torch.Tensor,
            x1: torch.Tensor,
            x2: torch.Tensor,
            x3: torch.Tensor,
            x4: torch.Tensor,
            x5: torch.Tensor,
            x6: torch.Tensor,
        ):
            if branch_idx == 0:
                return branch_fn0(x1, x2, x3, x4, x5, x6)
            if branch_idx == 1:
                return branch_fn1(x1, x2, x3, x4, x5, x6)
            if branch_idx == 2:
                return branch_fn2(x1, x2, x3, x4, x5, x6)
            if branch_idx == 3:
                return branch_fn3(x1, x2, x3, x4, x5, x6)
            if branch_idx == 4:
                return branch_fn4(x1, x2, x3, x4, x5, x6)
            if branch_idx == 5:
                return branch_fn5(x1, x2, x3, x4, x5, x6)
            if branch_idx == 6:
                return branch_fn6(x1, x2, x3, x4, x5, x6)
            if branch_idx == 7:
                return branch_fn7(x1, x2, x3, x4, x5, x6)
            if branch_idx == 8:
                return branch_fn8(x1, x2, x3, x4, x5, x6)
            raise AssertionError(f"Invalid branch index {branch_idx.item()}")

        def _switch7(
            branch_idx: torch.Tensor,
            x1: torch.Tensor,
            x2: torch.Tensor,
            x3: torch.Tensor,
            x4: torch.Tensor,
            x5: torch.Tensor,
            x6: torch.Tensor,
            x7: torch.Tensor,
        ):
            if branch_idx == 0:
                return branch_fn0(x1, x2, x3, x4, x5, x6, x7)
            if branch_idx == 1:
                return branch_fn1(x1, x2, x3, x4, x5, x6, x7)
            if branch_idx == 2:
                return branch_fn2(x1, x2, x3, x4, x5, x6, x7)
            if branch_idx == 3:
                return branch_fn3(x1, x2, x3, x4, x5, x6, x7)
            if branch_idx == 4:
                return branch_fn4(x1, x2, x3, x4, x5, x6, x7)
            if branch_idx == 5:
                return branch_fn5(x1, x2, x3, x4, x5, x6, x7)
            if branch_idx == 6:
                return branch_fn6(x1, x2, x3, x4, x5, x6, x7)
            if branch_idx == 7:
                return branch_fn7(x1, x2, x3, x4, x5, x6, x7)
            if branch_idx == 8:
                return branch_fn8(x1, x2, x3, x4, x5, x6, x7)
            raise AssertionError(f"Invalid branch index {branch_idx.item()}")

        def _switch8(
            branch_idx: torch.Tensor,
            x1: torch.Tensor,
            x2: torch.Tensor,
            x3: torch.Tensor,
            x4: torch.Tensor,
            x5: torch.Tensor,
            x6: torch.Tensor,
            x7: torch.Tensor,
            x8: torch.Tensor,
        ):
            if branch_idx == 0:
                return branch_fn0(x1, x2, x3, x4, x5, x6, x7, x8)
            if branch_idx == 1:
                return branch_fn1(x1, x2, x3, x4, x5, x6, x7, x8)
            if branch_idx == 2:
                return branch_fn2(x1, x2, x3, x4, x5, x6, x7, x8)
            if branch_idx == 3:
                return branch_fn3(x1, x2, x3, x4, x5, x6, x7, x8)
            if branch_idx == 4:
                return branch_fn4(x1, x2, x3, x4, x5, x6, x7, x8)
            if branch_idx == 5:
                return branch_fn5(x1, x2, x3, x4, x5, x6, x7, x8)
            if branch_idx == 6:
                return branch_fn6(x1, x2, x3, x4, x5, x6, x7, x8)
            if branch_idx == 7:
                return branch_fn7(x1, x2, x3, x4, x5, x6, x7, x8)
            if branch_idx == 8:
                return branch_fn8(x1, x2, x3, x4, x5, x6, x7, x8)
            raise AssertionError(f"Invalid branch index {branch_idx.item()}")

        def _switch9(
            branch_idx: torch.Tensor,
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
            if branch_idx == 0:
                return branch_fn0(x1, x2, x3, x4, x5, x6, x7, x8, x9)
            if branch_idx == 1:
                return branch_fn1(x1, x2, x3, x4, x5, x6, x7, x8, x9)
            if branch_idx == 2:
                return branch_fn2(x1, x2, x3, x4, x5, x6, x7, x8, x9)
            if branch_idx == 3:
                return branch_fn3(x1, x2, x3, x4, x5, x6, x7, x8, x9)
            if branch_idx == 4:
                return branch_fn4(x1, x2, x3, x4, x5, x6, x7, x8, x9)
            if branch_idx == 5:
                return branch_fn5(x1, x2, x3, x4, x5, x6, x7, x8, x9)
            if branch_idx == 6:
                return branch_fn6(x1, x2, x3, x4, x5, x6, x7, x8, x9)
            if branch_idx == 7:
                return branch_fn7(x1, x2, x3, x4, x5, x6, x7, x8, x9)
            if branch_idx == 8:
                return branch_fn8(x1, x2, x3, x4, x5, x6, x7, x8, x9)
            raise AssertionError(f"Invalid branch index {branch_idx.item()}")

        switch_dict = {
            1: _switch1,
            2: _switch2,
            3: _switch3,
            4: _switch4,
            5: _switch5,
            6: _switch6,
            7: _switch7,
            8: _switch8,
            9: _switch9,
        }
        assert len(original_args) <= len(
            switch_dict
        ), f"At most {len(switch_dict)} arguments are supported, got {len(original_args)}"
        compiled_switch = torch.jit.script(switch_dict[len(original_args)])
        return compiled_switch

    @torch.jit.ignore
    def _compile_state_switch_fn(self, original_args: Tuple[torch.Tensor, ...]):
        # use state for in-place modifications
        assert len(self.branch_fns) <= 9, f"At most 9 branches are supported, got {len(self.branch_fns)}"
        state_branch_fns: List[UseStateFunc] = []
        branch_fns: List[torch.jit.ScriptFunction] = []
        for branch_fn in self.branch_fns:
            state_branch_fns.append(use_state(branch_fn, is_generator=False))
            branch_fns.append(
                jit(
                    state_branch_fns[-1],
                    trace=True,
                    lazy=False,
                    example_inputs=(state_branch_fns[-1].init_state(False),) + original_args,
                )
            )
        state_branch_fns = tuple(state_branch_fns)
        # set local functions
        branch_fn0, branch_fn1, *_rest = branch_fns
        if len(_rest) > 0:
            branch_fn2, *_rest = _rest
        else:
            branch_fn2 = branch_fn0
        if len(_rest) > 0:
            branch_fn3, *_rest = _rest
        else:
            branch_fn3 = branch_fn0
        if len(_rest) > 0:
            branch_fn4, *_rest = _rest
        else:
            branch_fn4 = branch_fn0
        if len(_rest) > 0:
            branch_fn5, *_rest = _rest
        else:
            branch_fn5 = branch_fn0
        if len(_rest) > 0:
            branch_fn6, *_rest = _rest
        else:
            branch_fn6 = branch_fn0
        if len(_rest) > 0:
            branch_fn7, *_rest = _rest
        else:
            branch_fn7 = branch_fn0
        if len(_rest) > 0:
            branch_fn8, *_rest = _rest
        else:
            branch_fn8 = branch_fn0

        def _switch1(state: Dict[str, torch.Tensor], branch_idx: torch.Tensor, x1: torch.Tensor):
            if branch_idx == 0:
                return branch_fn0(state, x1)
            if branch_idx == 1:
                return branch_fn1(state, x1)
            if branch_idx == 2:
                return branch_fn2(state, x1)
            if branch_idx == 3:
                return branch_fn3(state, x1)
            if branch_idx == 4:
                return branch_fn4(state, x1)
            if branch_idx == 5:
                return branch_fn5(state, x1)
            if branch_idx == 6:
                return branch_fn6(state, x1)
            if branch_idx == 7:
                return branch_fn7(state, x1)
            if branch_idx == 8:
                return branch_fn8(state, x1)
            raise AssertionError(f"Invalid branch index {branch_idx.item()}")

        def _switch2(
            state: Dict[str, torch.Tensor],
            branch_idx: torch.Tensor,
            x1: torch.Tensor,
            x2: torch.Tensor,
        ):
            if branch_idx == 0:
                return branch_fn0(state, x1, x2)
            if branch_idx == 1:
                return branch_fn1(state, x1, x2)
            if branch_idx == 2:
                return branch_fn2(state, x1, x2)
            if branch_idx == 3:
                return branch_fn3(state, x1, x2)
            if branch_idx == 4:
                return branch_fn4(state, x1, x2)
            if branch_idx == 5:
                return branch_fn5(state, x1, x2)
            if branch_idx == 6:
                return branch_fn6(state, x1, x2)
            if branch_idx == 7:
                return branch_fn7(state, x1, x2)
            if branch_idx == 8:
                return branch_fn8(state, x1, x2)
            raise AssertionError(f"Invalid branch index {branch_idx.item()}")

        def _switch3(
            state: Dict[str, torch.Tensor],
            branch_idx: torch.Tensor,
            x1: torch.Tensor,
            x2: torch.Tensor,
            x3: torch.Tensor,
        ):
            if branch_idx == 0:
                return branch_fn0(state, x1, x2, x3)
            if branch_idx == 1:
                return branch_fn1(state, x1, x2, x3)
            if branch_idx == 2:
                return branch_fn2(state, x1, x2, x3)
            if branch_idx == 3:
                return branch_fn3(state, x1, x2, x3)
            if branch_idx == 4:
                return branch_fn4(state, x1, x2, x3)
            if branch_idx == 5:
                return branch_fn5(state, x1, x2, x3)
            if branch_idx == 6:
                return branch_fn6(state, x1, x2, x3)
            if branch_idx == 7:
                return branch_fn7(state, x1, x2, x3)
            if branch_idx == 8:
                return branch_fn8(state, x1, x2, x3)
            raise AssertionError(f"Invalid branch index {branch_idx.item()}")

        def _switch4(
            state: Dict[str, torch.Tensor],
            branch_idx: torch.Tensor,
            x1: torch.Tensor,
            x2: torch.Tensor,
            x3: torch.Tensor,
            x4: torch.Tensor,
        ):
            if branch_idx == 0:
                return branch_fn0(state, x1, x2, x3, x4)
            if branch_idx == 1:
                return branch_fn1(state, x1, x2, x3, x4)
            if branch_idx == 2:
                return branch_fn2(state, x1, x2, x3, x4)
            if branch_idx == 3:
                return branch_fn3(state, x1, x2, x3, x4)
            if branch_idx == 4:
                return branch_fn4(state, x1, x2, x3, x4)
            if branch_idx == 5:
                return branch_fn5(state, x1, x2, x3, x4)
            if branch_idx == 6:
                return branch_fn6(state, x1, x2, x3, x4)
            if branch_idx == 7:
                return branch_fn7(state, x1, x2, x3, x4)
            if branch_idx == 8:
                return branch_fn8(state, x1, x2, x3, x4)
            raise AssertionError(f"Invalid branch index {branch_idx.item()}")

        def _switch5(
            state: Dict[str, torch.Tensor],
            branch_idx: torch.Tensor,
            x1: torch.Tensor,
            x2: torch.Tensor,
            x3: torch.Tensor,
            x4: torch.Tensor,
            x5: torch.Tensor,
        ):
            if branch_idx == 0:
                return branch_fn0(state, x1, x2, x3, x4, x5)
            if branch_idx == 1:
                return branch_fn1(state, x1, x2, x3, x4, x5)
            if branch_idx == 2:
                return branch_fn2(state, x1, x2, x3, x4, x5)
            if branch_idx == 3:
                return branch_fn3(state, x1, x2, x3, x4, x5)
            if branch_idx == 4:
                return branch_fn4(state, x1, x2, x3, x4, x5)
            if branch_idx == 5:
                return branch_fn5(state, x1, x2, x3, x4, x5)
            if branch_idx == 6:
                return branch_fn6(state, x1, x2, x3, x4, x5)
            if branch_idx == 7:
                return branch_fn7(state, x1, x2, x3, x4, x5)
            if branch_idx == 8:
                return branch_fn8(state, x1, x2, x3, x4, x5)
            raise AssertionError(f"Invalid branch index {branch_idx.item()}")

        def _switch6(
            state: Dict[str, torch.Tensor],
            branch_idx: torch.Tensor,
            x1: torch.Tensor,
            x2: torch.Tensor,
            x3: torch.Tensor,
            x4: torch.Tensor,
            x5: torch.Tensor,
            x6: torch.Tensor,
        ):
            if branch_idx == 0:
                return branch_fn0(state, x1, x2, x3, x4, x5, x6)
            if branch_idx == 1:
                return branch_fn1(state, x1, x2, x3, x4, x5, x6)
            if branch_idx == 2:
                return branch_fn2(state, x1, x2, x3, x4, x5, x6)
            if branch_idx == 3:
                return branch_fn3(state, x1, x2, x3, x4, x5, x6)
            if branch_idx == 4:
                return branch_fn4(state, x1, x2, x3, x4, x5, x6)
            if branch_idx == 5:
                return branch_fn5(state, x1, x2, x3, x4, x5, x6)
            if branch_idx == 6:
                return branch_fn6(state, x1, x2, x3, x4, x5, x6)
            if branch_idx == 7:
                return branch_fn7(state, x1, x2, x3, x4, x5, x6)
            if branch_idx == 8:
                return branch_fn8(state, x1, x2, x3, x4, x5, x6)
            raise AssertionError(f"Invalid branch index {branch_idx.item()}")

        def _switch7(
            state: Dict[str, torch.Tensor],
            branch_idx: torch.Tensor,
            x1: torch.Tensor,
            x2: torch.Tensor,
            x3: torch.Tensor,
            x4: torch.Tensor,
            x5: torch.Tensor,
            x6: torch.Tensor,
            x7: torch.Tensor,
        ):
            if branch_idx == 0:
                return branch_fn0(state, x1, x2, x3, x4, x5, x6, x7)
            if branch_idx == 1:
                return branch_fn1(state, x1, x2, x3, x4, x5, x6, x7)
            if branch_idx == 2:
                return branch_fn2(state, x1, x2, x3, x4, x5, x6, x7)
            if branch_idx == 3:
                return branch_fn3(state, x1, x2, x3, x4, x5, x6, x7)
            if branch_idx == 4:
                return branch_fn4(state, x1, x2, x3, x4, x5, x6, x7)
            if branch_idx == 5:
                return branch_fn5(state, x1, x2, x3, x4, x5, x6, x7)
            if branch_idx == 6:
                return branch_fn6(state, x1, x2, x3, x4, x5, x6, x7)
            if branch_idx == 7:
                return branch_fn7(state, x1, x2, x3, x4, x5, x6, x7)
            if branch_idx == 8:
                return branch_fn8(state, x1, x2, x3, x4, x5, x6, x7)
            raise AssertionError(f"Invalid branch index {branch_idx.item()}")

        def _switch8(
            state: Dict[str, torch.Tensor],
            branch_idx: torch.Tensor,
            x1: torch.Tensor,
            x2: torch.Tensor,
            x3: torch.Tensor,
            x4: torch.Tensor,
            x5: torch.Tensor,
            x6: torch.Tensor,
            x7: torch.Tensor,
            x8: torch.Tensor,
        ):
            if branch_idx == 0:
                return branch_fn0(state, x1, x2, x3, x4, x5, x6, x7, x8)
            if branch_idx == 1:
                return branch_fn1(state, x1, x2, x3, x4, x5, x6, x7, x8)
            if branch_idx == 2:
                return branch_fn2(state, x1, x2, x3, x4, x5, x6, x7, x8)
            if branch_idx == 3:
                return branch_fn3(state, x1, x2, x3, x4, x5, x6, x7, x8)
            if branch_idx == 4:
                return branch_fn4(state, x1, x2, x3, x4, x5, x6, x7, x8)
            if branch_idx == 5:
                return branch_fn5(state, x1, x2, x3, x4, x5, x6, x7, x8)
            if branch_idx == 6:
                return branch_fn6(state, x1, x2, x3, x4, x5, x6, x7, x8)
            if branch_idx == 7:
                return branch_fn7(state, x1, x2, x3, x4, x5, x6, x7, x8)
            if branch_idx == 8:
                return branch_fn8(state, x1, x2, x3, x4, x5, x6, x7, x8)
            raise AssertionError(f"Invalid branch index {branch_idx.item()}")

        def _switch9(
            state: Dict[str, torch.Tensor],
            branch_idx: torch.Tensor,
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
            if branch_idx == 0:
                return branch_fn0(state, x1, x2, x3, x4, x5, x6, x7, x8, x9)
            if branch_idx == 1:
                return branch_fn1(state, x1, x2, x3, x4, x5, x6, x7, x8, x9)
            if branch_idx == 2:
                return branch_fn2(state, x1, x2, x3, x4, x5, x6, x7, x8, x9)
            if branch_idx == 3:
                return branch_fn3(state, x1, x2, x3, x4, x5, x6, x7, x8, x9)
            if branch_idx == 4:
                return branch_fn4(state, x1, x2, x3, x4, x5, x6, x7, x8, x9)
            if branch_idx == 5:
                return branch_fn5(state, x1, x2, x3, x4, x5, x6, x7, x8, x9)
            if branch_idx == 6:
                return branch_fn6(state, x1, x2, x3, x4, x5, x6, x7, x8, x9)
            if branch_idx == 7:
                return branch_fn7(state, x1, x2, x3, x4, x5, x6, x7, x8, x9)
            if branch_idx == 8:
                return branch_fn8(state, x1, x2, x3, x4, x5, x6, x7, x8, x9)
            raise AssertionError(f"Invalid branch index {branch_idx.item()}")

        switch_dict = {
            1: _switch1,
            2: _switch2,
            3: _switch3,
            4: _switch4,
            5: _switch5,
            6: _switch6,
            7: _switch7,
            8: _switch8,
            9: _switch9,
        }
        assert len(original_args) <= len(
            switch_dict
        ), f"At most {len(switch_dict)} arguments are supported, got {len(original_args)}"
        compiled_switch = torch.jit.script(switch_dict[len(original_args)])
        return compiled_switch, state_branch_fns

    @trace_impl(switch)
    def trace_switch(self, branch_idx: torch.Tensor | Dict[str, torch.Tensor], *x: torch.Tensor):
        if self.stateful_functions:
            state: Dict[str, torch.Tensor] = branch_idx
            branch_idx = x[0]
            x = x[1:]
        else:
            state = None
        key = tuple((a.ndim, a.dtype, a.device) for a in (branch_idx,) + x)
        if state is not None:
            if key in self._cache_compiled_switch:
                compiled_switch = self._cache_compiled_switch[key]
            else:
                compiled_switch, _ = self._compile_state_switch_fn(x)
                self._cache_compiled_switch[key] = compiled_switch
            res = compiled_switch(state, branch_idx, *x)
        else:
            if key in self._cache_compiled_switch:
                compiled_switch = self._cache_compiled_switch[key]
            else:
                compiled_switch = self._compile_switch_fn(x)
                self._cache_compiled_switch[key] = compiled_switch
            res = compiled_switch(branch_idx, *x)
        return res

    @vmap_impl(switch)
    def vmap_switch(self, branch_idx: torch.Tensor | Dict[str, torch.Tensor], *x: torch.Tensor):
        # cannot dynamic dispatch for vmap, change to torch.where
        # use_state to support in-place modifications
        if self.stateful_functions:
            state: Dict[str, torch.Tensor] = branch_idx
            branch_idx = x[0]
            x = x[1:]
            ori_state = {k: _param_clone(v) for k, v in state.items()}
        else:
            state = None
        branch_results: List[torch.Tensor | Tuple[torch.Tensor, ...] | None] = []
        if state is not None:
            final_states: List[Dict[str, torch.Tensor]] = []
        for branch_fn in self.branch_fns:
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
            final_output = _switch(branch_idx, branch_results)
        elif isinstance(branch_results[0], Sequence):
            final_output = []
            for results in zip(*branch_results):
                final_output.append(_switch(branch_idx, results))
        else:
            raise ValueError("The type of returns of branches should be the same.")
        if state is None:
            return final_output
        else:
            # set conditional state
            state_out: Dict[str, torch.Tensor] = {}
            for k, v in ori_state.items():
                final_state_tensors = [s.get(k, v) for s in final_states]
                state_out[k] = _switch(branch_idx, final_state_tensors)
            # return
            return state_out, final_output
