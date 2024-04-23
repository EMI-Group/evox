import numpy as np
import jax
from jax.tree_util import tree_map


def x32_func_call(func):
    def inner_func(*args, **kwargs):
        return to_x32_if_needed(func(*args, **kwargs))

    return inner_func


def to_x32_if_needed(values):
    if jax.config.jax_enable_x64:
        # we have 64-bit enabled, so nothing to do
        return values

    def to_x32(value):
        if value.dtype == np.float64:
            return value.astype(np.float32)
        elif value.dtype == np.int64:
            return value.astype(np.int32)
        else:
            return value

    return tree_map(to_x32, values)
