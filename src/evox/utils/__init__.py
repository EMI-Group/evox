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
    "TracingWhile",
    "TracingCond",
    "TracingSwitch",
    "ParamsAndVector",
    "lexsort",
    "nanmin",
    "nanmax",
]

from .control_flow import TracingCond, TracingSwitch, TracingWhile
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
from .parameters_and_vector import ParamsAndVector


################### NOTICE ###################
#
# 1. The functions in this module are all for JIT operator fusion, JIT tracing, or vectorized-map since their original implementations are not supported.
# 2. When using `core.vmap`, all input tensors are assumed to be float tensors. If integer tensors are used, please use `core.vmap(..., trace=False)` and manually JIT it afterward using `core.jit(..., trace=True, example_inputs=(...))` or add type hint of `torch.IntTensor`.
# 3. Python's control flow statements cannot be vector-mapped directly, please use the function in this module instead.
# 4. DO NOT directly use `torch.jit.script` to JIT `torch.vmap` functions. You may get unexpected results without any warning.
#
################# END NOTICE #################
