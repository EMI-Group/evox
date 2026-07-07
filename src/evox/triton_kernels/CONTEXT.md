# Triton Kernels

## Intent
Optional GPU/accelerator-accelerated operations using Triton. When Triton is installed and the target backend is available, custom operations dispatch to hand-written Triton kernels. Supported backends include NVIDIA CUDA, AMD ROCm (aliased under the `"cuda"` device type), and Ascend NPU (`"npu"`, when registered). When Triton is not available or the device is unsupported, operations fall back to pure PyTorch implementations automatically via PyTorch's dispatcher.

## API Surface
- `has_triton()` — Check if Triton is importable.
- `triton_supports_device(device)` — Check if Triton can run on a given device (checks the registered device-type set).
- `triton_device_types()` — Return the frozenset of registered Triton device types (defaults to `{"cuda"}`).
- `register_triton_device_type(device_type)` — Register an additional device type (e.g. `"npu"`) for Triton kernel support.
- `register_triton_op(...)` — Decorator to register an op with PyTorch fallback + Triton kernel(s). Accepts an optional `triton_device_types` parameter (a single string or sequence of strings) to extend or restrict which backends receive a Triton kernel; when `None`, defaults to the globally registered device types.
- `kernels/` — Submodule containing individual Triton kernel implementations.

## Constraints
- **Triton is optional**: The module MUST import and work correctly without Triton installed. All Triton imports must be guarded by `has_triton()`.
- **Configurable device types**: Triton kernels are registered for a configurable set of device types, defaulting to `{"cuda"}` (covers NVIDIA CUDA and AMD ROCm). The set is extensible via `register_triton_device_type()` (e.g. to add Ascend NPU `"npu"`). PyTorch fallback handles any device not covered, including CPU and MPS.
- **Lazy registration**: Triton kernels are registered at import time only if Triton is available, but kernel compilation is lazy (JIT-compiled on first call).
- **Follow existing patterns**: Use `register_vmap_op` under the hood; ops live in the `evox::` namespace.
- **Pure PyTorch fallbacks**: Fallback functions must be pure PyTorch (no NumPy) for GPU compatibility and compile-friendliness.

## Routing Table
| Area | Path | Description |
|---|---|---|
| Kernel implementations | `kernels/` | Individual Triton kernel files, each demonstrating the full registration pattern |
| Fused element-wise ops | `kernels/fused_add.py` | Example fused addition kernel (reference pattern) |
| Philox PRNG | `kernels/philox.py` | Philox4x32-10 deterministic PRNG: `philox_uniform` ([0,1) floats), `philox_normal` (Box-Muller standard normal). Triton CUDA kernel + pure-PyTorch fallback |
| LoRA noise utilities | `kernels/lora_noise.py` | Pure-PyTorch LoRA perturbation helpers built on `philox_normal`: `compute_counter_offsets`, `generate_lora_factors`, `lora_delta_output`, `lora_gradient` |
