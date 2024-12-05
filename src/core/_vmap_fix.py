from typing import Any, List, Tuple

import torch
from torch._C._functorch import (_add_batch_dim as add_batch_dim, maybe_get_bdim as get_batch_dim,
                                 maybe_get_level as get_level, maybe_current_level as current_level,
                                 is_batchedtensor as is_batched_tensor, get_unwrapped,
                                 _unwrap_batched as unwrap_batched, _vmap_increment_nesting as
                                 vmap_increment_nesting, _vmap_decrement_nesting as
                                 vmap_decrement_nesting)

from torch.utils._pytree import tree_unflatten
import torch._functorch.vmap


def _transform_in_dim(in_dim: int | Tuple[int, ...], bached: torch.Tensor, original: torch.Tensor):
    if not isinstance(in_dim, tuple):
        in_dim = (in_dim,)
    shape = original.size()
    shape = tuple(s for i, s in enumerate(shape) if i not in in_dim)
    bached.size = lambda i=None: shape if i is None else shape[i]


def _create_batched_inputs(flat_in_dims: List[Any], flat_args: List[Any], vmap_level: int,
                           args_spec) -> Tuple:
    # See NOTE [Ignored _remove_batch_dim, _add_batch_dim]
    batched_inputs = [
        arg if in_dim is None else add_batch_dim(arg, in_dim, vmap_level)
        for in_dim, arg in zip(flat_in_dims, flat_args)
    ]
    for batched, arg, in_dim in zip(batched_inputs, flat_args, flat_in_dims):
        if isinstance(batched, torch.Tensor):
            _transform_in_dim(in_dim, batched, arg)
    return tree_unflatten(batched_inputs, args_spec)


torch._functorch.vmap._create_batched_inputs = _create_batched_inputs
