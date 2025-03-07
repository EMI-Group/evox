__all__ = [
    "switch",
    "clamp",
    "clamp_int",
    "clamp_float",
    "clip",
    "maximum",
    "minimum",
    "maximum_float",
    "minimum_float",
    "maximum_int",
    "minimum_int",
    "VmapInfo",
    "register_vmap_op",
    "ParamsAndVector",
    "lexsort",
    "nanmin",
    "nanmax",
    "tree_flatten",
    "tree_unflatten",
]

from .jit_fix_operator import (
    clamp,
    clamp_float,
    clamp_int,
    clip,
    lexsort,
    maximum,
    maximum_float,
    maximum_int,
    minimum,
    minimum_float,
    minimum_int,
    nanmax,
    nanmin,
    switch,
)
from .op_register import VmapInfo, register_vmap_op
from .parameters_and_vector import ParamsAndVector
from .re_export import tree_flatten, tree_unflatten


################### NOTICE ###################
#
# 1. The functions in this module are all for compiler operator fusion, or vectorized-map since their original implementations are not supported in operator fusion yet or not present in PyTorch and may cause performance loss. However, these implmentations have their own limitations, please read the docstrings for details.
# 2. Python's control flow statements cannot be vector-mapped directly, please use `torch.cond` for `if-else` statement and `torch.while_loop` for `while` and `for` loops. If you want to vectorized-map a loop, please follow in the instruction on the documentation.
#
################# END NOTICE #################
