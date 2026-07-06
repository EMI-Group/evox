# Triton Kernels

## Intent
Optional GPU-accelerated operations using Triton. When Triton is installed and CUDA is available, custom operations dispatch to hand-written Triton kernels. When Triton is not available, operations fall back to pure PyTorch implementations automatically via PyTorch's dispatcher.

## API Surface
- `has_triton()` — Check if Triton is importable.
- `triton_supports_device(device)` — Check if Triton can run on a given device.
- `register_triton_op(...)` — Decorator to register an op with PyTorch fallback + Triton CUDA kernel.
- `kernels/` — Submodule containing individual Triton kernel implementations.

## Constraints
- **Triton is optional**: The module MUST import and work correctly without Triton installed. All Triton imports must be guarded by `has_triton()`.
- **CUDA-only kernels**: Triton kernels only run on CUDA devices. PyTorch fallback handles CPU, MPS, and all other devices.
- **Lazy registration**: Triton kernels are registered at import time only if Triton is available, but kernel compilation is lazy (JIT-compiled on first call).
- **Follow existing patterns**: Use `register_vmap_op` under the hood; ops live in the `evox::` namespace.
- **Pure PyTorch fallbacks**: Fallback functions must be pure PyTorch (no NumPy) for GPU compatibility and compile-friendliness.

## Routing Table
| Area | Path | Description |
|---|---|---|
| Kernel implementations | `kernels/` | Individual Triton kernel files, each demonstrating the full registration pattern |
