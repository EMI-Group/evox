__all__ = ["cond", "switch", "clamp", "clip", "maximum", "minimum", "TracingWhileLoop"]

from .jit_fix_operator import cond, switch, clamp, clip, maximum, minimum
from .while_loop import TracingWhileLoop


################### NOTICE ###################
#
# 1. The functions in this module are all for JIT operator fusion since their original implementations are not supported in fusion.
# 2. When using `core.vmap`, all input tensors are assumed to be float tensors. If integer tensors are used, please use `core.vmap(..., trace=False)` and manually JIT it afterward using `core.jit(..., trace=True, example_inputs=(...))`.
# 3. When using `core.vmap`, two batched tensors cannot directly slice-gathered like `tensor_a[tensor_idx]`. Please use `torch.index_select(tensor_a, 0, tensor_idx)` instead.
# 4. Python's while loops cannot be vector-mapped directly, please use the function in this module instead.
# 5. DO NOT directly use `torch.jit.script` to JIT `torch.vmap` functions. You may get unexpected results without any warning.
#
################# END NOTICE #################
