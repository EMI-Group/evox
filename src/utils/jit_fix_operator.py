import torch


################### NOTICE ###################
#
# 1. The functions in this module are all for JIT operator fusion since their original implementations are not supported in fusion.
# 2. When using `core.vmap`, all input tensors are assumed to be float tensors. If integer tensors are used, please use `core.vmap(..., trace=False)` and manually JIT it afterward using `core.jit(..., trace=True, example_inputs=(...))`.
# 3. When using `core.vmap`, two batched tensors cannot directly slice-gathered like `tensor_a[tensor_idx]`. Please use `torch.index_select(tensor_a, 0, tensor_idx)` instead.
# 4. Python's while loops cannot be vector-mapped directly, please use the function in this module instead.
#
################# END NOTICE #################


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