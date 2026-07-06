from typing import Sequence

import torch

from ..utils.op_register import register_vmap_op
from .backend import has_triton, triton_device_types as _get_triton_device_types


def register_triton_op(
    *,
    fake_fn,
    triton_fn,
    vmap_fn=None,
    fake_vmap_fn=None,
    vmap_wrap_inputs=None,
    vmap_out_dims=0,
    max_vmap_level=None,
    name=None,
    mutates_args=(),
    device_types=None,
    triton_device_types: str | Sequence[str] | None = None,
    schema=None,
):
    """Register an operator with a PyTorch fallback and an optional Triton CUDA kernel.

    This decorator wraps :func:`evox.utils.register_vmap_op`, registering the decorated
    function (``fallback_fn``) as the default implementation with fake / vmap support,
    and additionally registering ``triton_fn`` as a CUDA-specific kernel via
    :func:`torch.library.register_kernel`.

    PyTorch's dispatcher automatically selects the correct backend at call time:

    - **CUDA tensors** → Triton kernel (``triton_fn``)
    - **CPU / MPS / other tensors** → PyTorch fallback (the decorated function)

    If Triton is not installed, only the PyTorch fallback is registered and the
    operation works on all devices without Triton.

    :param fake_fn: The fake (abstract evaluation) function for the op. Required.
    :param triton_fn: The Triton CUDA kernel launcher. Called with the same arguments
        as the decorated function when dispatched on CUDA. Must return outputs matching
        the fake function's shapes/dtypes.
    :param vmap_fn: Optional vmap implementation. See :func:`register_vmap_op`.
    :param fake_vmap_fn: Optional fake vmap function. See :func:`register_vmap_op`.
    :param vmap_wrap_inputs: Optional input wrapper for vmap. See :func:`register_vmap_op`.
    :param vmap_out_dims: Output vmap dimensions. See :func:`register_vmap_op`.
    :param max_vmap_level: Maximum vmap nesting level. See :func:`register_vmap_op`.
    :param name: Custom op name. Default ``"evox::_custom_op_" + fn.__name__``.
    :param mutates_args: Args mutated by the op. See :func:`register_vmap_op`.
    :param device_types: Supported device types for the op registration.
    :param triton_device_types: Device type(s) for which the Triton kernel is
        registered via :func:`torch.library.register_kernel`. When ``None``
        (default), uses the globally registered device types
        (:func:`~evox.triton_kernels.backend.triton_device_types`, default
        ``{"cuda"}``). Accepts a single string (e.g. ``"npu"``) or a sequence
        of strings (e.g. ``["cuda", "npu"]``) to extend or restrict coverage.
    :param schema: Op schema string.

    ## Example

    ```python
    @triton.jit
    def _add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x + y, mask=mask)

    def _triton_add(x, y):
        out = torch.empty_like(x)
        n = x.numel()
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        _add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)
        return out

    def _add_fake(x, y):
        return torch.empty_like(x)

    @register_triton_op(fake_fn=_add_fake, triton_fn=_triton_add)
    def fused_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y
    ```
    """
    def decorator(fallback_fn):
        # Step 1: Register the PyTorch fallback as the default op + fake + vmap
        registered = register_vmap_op(
            fallback_fn,
            fake_fn=fake_fn,
            vmap_fn=vmap_fn,
            fake_vmap_fn=fake_vmap_fn,
            vmap_wrap_inputs=vmap_wrap_inputs,
            vmap_out_dims=vmap_out_dims,
            max_vmap_level=max_vmap_level,
            name=name,
            mutates_args=mutates_args,
            device_types=device_types,
            schema=schema,
        )

        # Step 2: Determine the op name (must match register_vmap_op's naming logic)
        op_name = name if name is not None else "evox::_custom_op_" + fallback_fn.__name__

        # Step 3: Register the Triton kernel(s) for the supported device types,
        # but only if Triton is available. By default this covers the globally
        # registered device types (e.g. "cuda"), but callers may pass an explicit
        # ``triton_device_types`` to extend or restrict coverage (e.g. "npu").
        if has_triton():
            if triton_device_types is None:
                resolved_types = _get_triton_device_types()
            elif isinstance(triton_device_types, str):
                resolved_types = [triton_device_types]
            else:
                resolved_types = triton_device_types
            for dt in resolved_types:
                try:
                    torch.library.register_kernel(op_name, dt, triton_fn)
                except RuntimeError:
                    # The device type is not recognized by this PyTorch build
                    # (e.g. "npu" without torch-npu installed). Silently skip so
                    # the module imports cleanly on all platforms; the kernel
                    # will register when the platform-specific backend is
                    # available, with the PyTorch fallback used in the meantime.
                    continue

        return registered

    return decorator
