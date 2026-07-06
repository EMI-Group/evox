"""Example fused element-wise addition kernel.

This module demonstrates the full Triton kernel integration pattern:

1. Define the ``@triton.jit`` kernel function.
2. Define a launcher function that sets up the grid and calls the kernel.
3. Define a PyTorch fallback that runs on all devices (CPU, MPS, etc.).
4. Define a fake (abstract evaluation) function for ``torch.compile`` tracing.
5. Register everything via :func:`~evox.triton_kernels.register_triton_op`.

At call time, PyTorch's dispatcher routes CUDA tensors to the Triton kernel
and all other tensors to the PyTorch fallback automatically.
"""

import torch

from ..backend import has_triton
from ..op_register import register_triton_op

if has_triton():
    import triton
    import triton.language as tl


    @triton.jit
    def _add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        """Triton kernel for element-wise addition."""
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x + y, mask=mask)


def _triton_fused_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Launch the Triton addition kernel."""
    out = torch.empty_like(x)
    n_elements = x.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=1024)
    return out


def _fused_add_fake(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Fake (abstract evaluation) function for torch.compile tracing."""
    return torch.empty_like(x)


@register_triton_op(fake_fn=_fused_add_fake, triton_fn=_triton_fused_add)
def fused_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Element-wise addition of two tensors with the same shape.

    Uses a fused Triton kernel on CUDA devices and falls back to PyTorch's
    native ``+`` operator on all other devices (CPU, MPS, etc.).

    :param x: First input tensor.
    :param y: Second input tensor, must have the same shape as ``x``.
    :return: Element-wise sum ``x + y``.
    """
    return x + y
