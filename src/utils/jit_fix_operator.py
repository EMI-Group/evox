from functools import wraps
from typing import Callable, Dict, List, Tuple, TypeVarTuple

import torch

from ..core import trace_impl, vmap_impl, jit_class, vmap, jit, ModuleBase
from ..core import _vmap_fix


################### NOTICE ###################
#
# 1. The functions in this module are all for JIT operator fusion since their original implementations are not supported in fusion.
# 2. When using `core.vmap`, all input tensors are assumed to be float tensors. If integer tensors are used, please use `core.vmap(..., trace=False)` and manually JIT it afterward using `core.jit(..., trace=True, example_inputs=(...))`.
# 3. When using `core.vmap`, two batched tensors cannot directly slice-gathered like `tensor_a[tensor_idx]`. Please use `torch.index_select(tensor_a, 0, tensor_idx)` instead.
# 4. Python's while loops cannot be vector-mapped directly, please use the function in this module instead.
# 5. DO NOT directly use `torch.jit.script` to JIT `torch.vmap` functions. You may get unexpected results without any warning.
#
################# END NOTICE #################


@jit_class
class TracingWhileLoop(ModuleBase):

    def __init__(
        self,
        cond: Callable[[*Tuple[torch.Tensor, ...]], torch.Tensor],
        body: Callable[[*Tuple[torch.Tensor, ...]], Tuple[torch.Tensor, ...]],
    ):
        super().__init__()
        self.cond = torch.jit.script(cond)
        self.body = torch.jit.script(body)
        self._cond = cond
        self._body = body
        self._cache_compiled_vmap_loop: Dict[Tuple[Tuple[int, ...], int], torch.jit.ScriptFunction] = {}

    @torch.jit.ignore
    def loop(self, *x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        while self.cond(*x):
            x = self.body(*x)
        return x

    @trace_impl(loop)
    def trace_loop(self, *x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        cond: torch.jit.ScriptFunction = self.cond
        body: torch.jit.ScriptFunction = self.body

        def _loop1(x1: torch.Tensor) -> torch.Tensor:
            while cond(x1):
                x1 = body(x1)
            return x1

        def _loop2(x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            while cond(x1, x2):
                x1, x2 = body(x1, x2)
            return x1, x2

        def _loop3(
            x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            while cond(x1, x2, x3):
                x1, x2, x3 = body(x1, x2, x3)
            return x1, x2, x3

        def _loop4(
            x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, x4: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            while cond(x1, x2, x3, x4):
                x1, x2, x3, x4 = body(x1, x2, x3, x4)
            return x1, x2, x3, x4

        def _loop5(
            x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, x4: torch.Tensor, x5: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            while cond(x1, x2, x3, x4, x5):
                x1, x2, x3, x4, x5 = body(x1, x2, x3, x4, x5)
            return x1, x2, x3, x4, x5

        loops_dict = {1: _loop1, 2: _loop2, 3: _loop3, 4: _loop4, 5: _loop5}

        assert len(x) <= len(
            loops_dict
        ), f"At most {len(loops_dict)} arguments are supported, got {len(x)}"
        compiled_loop = torch.jit.script(loops_dict[len(x)])
        return compiled_loop(*x)

    @torch.jit.ignore
    def _compile_vmap_loop_fn(self, original_args: Tuple[torch.Tensor, ...], vmap_dims: Tuple[Tuple[int, ...], ...]):
        # vmap
        vmap_cond = self._cond
        vmap_body = self._body
        for d in zip(*vmap_dims):
            vmap_cond = vmap(vmap_cond, in_dims=d, trace=False)
            vmap_body = vmap(vmap_body, in_dims=d, out_dims=d, trace=False)

        def _expand_vmap_dim(
            vmap_dims: Tuple[int, ...], size: Tuple[int, ...], a: torch.Tensor
        ) -> torch.Tensor:
            size = list(size)
            for i, _ in enumerate(size):
                if i not in vmap_dims:
                    size[i] = 1
            return a.view(*size)

        def _vmap_cond_fn(*xs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
            cond_res = vmap_cond(*xs)
            cond_res = tuple(_expand_vmap_dim(d, a.size(), cond_res) for d, a in zip(vmap_dims, xs))
            return cond_res

        # JIT
        body = jit(vmap_body, trace=True, example_inputs=original_args)
        cond = jit(_vmap_cond_fn, trace=True, example_inputs=original_args)

        def _loop1(x1: torch.Tensor) -> torch.Tensor:
            cond_res = cond(x1)[0]
            while cond_res.any():
                x1_new = body(x1)
                x1 = torch.where(cond_res, x1_new, x1)
                cond_res = cond(x1)
            return x1

        def _loop2(x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            cond_res1, cond_res2 = cond(x1, x2)
            while cond_res1.any():
                x1_new, x2_new = body(x1, x2)
                x1 = torch.where(cond_res1, x1_new, x1)
                x2 = torch.where(cond_res2, x2_new, x2)
                cond_res1, cond_res2 = cond(x1, x2)
            return x1, x2

        def _loop3(
            x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            cond_res1, cond_res2, cond_res3 = cond(x1, x2, x3)
            while cond_res1.any():
                x1_new, x2_new, x3_new = body(x1, x2, x3)
                x1 = torch.where(cond_res1, x1_new, x1)
                x2 = torch.where(cond_res2, x2_new, x2)
                x3 = torch.where(cond_res3, x3_new, x3)
                cond_res1, cond_res2, cond_res3 = cond(x1, x2, x3)
            return x1, x2, x3

        def _loop4(
            x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, x4: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        loops_dict = {1: _loop1, 2: _loop2, 3: _loop3, 4: _loop4, 5: _loop5}
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


def clamp(a: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
    """
    Clamp the values of the input tensor `a` to be within the given lower (`lb`) and upper (`ub`) bounds.

    This function ensures that each element of the tensor `a` is not less than the corresponding element
    of `lb` and not greater than the corresponding element of `ub`.

    Notice: This is a fix function for [`torch.clamp`](https://pytorch.org/docs/stable/generated/torch.clamp.html) since it is not supported in JIT operator fusion.

    Args:
        a (`torch.Tensor`): The input tensor to be clamped.
        lb (`torch.Tensor`): The lower bound tensor. Must be broadcastable to the shape of `a`.
        ub (`torch.Tensor`): The upper bound tensor. Must be broadcastable to the shape of `a`.

    Returns:
        `torch.Tensor`: A tensor where each element is clamped to be within the specified bounds.
    """
    lb = torch.relu(lb - a)
    ub = torch.relu(a - ub)
    return a + lb - ub


def clip(a: torch.Tensor) -> torch.Tensor:
    """
    Clip the values of the input tensor `a` to be within the range [0, 1].

    Notice: This function invokes `clamp(a, 0, 1)`.

    Args:
        a (`torch.Tensor`): The input tensor to be clipped.

    Returns:
        `torch.Tensor`: A tensor where each element is clipped to be within [0, 1].
    """
    return clamp(a, 0, 1)


def maximum(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Element-wise maximum of two input tensors `a` and `b`.

    Notice: This is a fix function for [`torch.maximum`](https://pytorch.org/docs/stable/generated/torch.maximum.html] since it is not supported in JIT operator fusion.

    Args:
        a (`torch.Tensor`): The first input tensor.
        b (`torch.Tensor`): The second input tensor.

    Returns:
        `torch.Tensor`: The element-wise maximum of `a` and `b`.
    """
    diff = torch.relu(b - a)
    return a + diff


def minimum(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Element-wise minimum of two input tensors `a` and `b`.

    Notice: This is a fix function for [`torch.minimum`](https://pytorch.org/docs/stable/generated/torch.minimum.html] since it is not supported in JIT operator fusion.

    Args:
        a (`torch.Tensor`): The first input tensor.
        b (`torch.Tensor`): The second input tensor.

    Returns:
        `torch.Tensor`: The element-wise minimum of `a` and `b`.
    """
    diff = torch.relu(a - b)
    return a - diff
