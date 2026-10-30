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
| Virtual noise utilities | `kernels/virtual_noise.py` | Full-density Gaussian noise fused into matmul (never materialized). `virtual_perturbed_linear` (forward: `Y[i]=X@(W+sigma*N_i)^T+...`), `virtual_weight_gradient`/`virtual_bias_gradient` (regenerate the *same* noise for gradient estimates), `compute_offsets` (simple cumulative flat-element offsets). Triton kernel uses `tl.randn`/`tl.rand`; PyTorch fallback uses a splitmix64-based deterministic PRNG shared by forward+gradient |

## Noise-Index Conventions (parity-critical)
- `compute_offsets(param_shapes)` → per-block **starting flat element index** in the dense parameter vector; each block advances by `prod(shape)`.
- `compute_counter_offsets(param_shapes, lora_rank)` → per-block **starting Philox counter** (one counter = 4 uint32 values); each block advances by `_ceil_div4(n_elements)` with `n_elements = shape[0]` for 1-D and `lora_rank*k + d*lora_rank` for ≥2-D (`d = prod(shape[:-1])`, `k = shape[-1]`).
- The two offset lists live in different units and are not interchangeable; a LoRA bias block starts at `counter_offsets[bias_idx]`, NOT at `weight_counter + out*in`.
- `virtual_perturbed_linear` derives the bias noise offset internally as `offset + out_features*in_features`, so callers pass only the weight block offset; this is only valid because `nn.Linear` `named_parameters()` yields weight then bias.
- `generate_lora_factors(seeds, weight_shape, rank, counter)` — the 4th parameter is named `counter` (there is no `counter_offset` parameter), and `sigma` is applied by callers, never inside.
- 1-D `weight_shape` makes `generate_lora_factors` return a `(pop_size, n)` flat noise tensor; ≥2-D returns a tuple `(A, B)` with `A: (pop, rank, k)` and `B: (pop, d, rank)`.

## Known Issues
- `virtual_noise` is not cross-backend numerically consistent: the fallback (splitmix64 hash + Box-Muller `cos`) and the Triton kernel (`tl.randn`, or `tl.rand` + Box-Muller) yield different values for the same `(seed, offset)`; only forward/gradient agreement *within* one backend is guaranteed (`philox`/LoRA helpers, by contrast, are bit-identical across backends).
- The `virtual_perturbed_linear` fallback materializes the whole `(pop_size, out*in)` weight-noise tensor before its per-individual loop.
- Triton launchers pass `seeds` as `int32` and build `int32` noise offsets, so seeds/offsets ≥ 2^31 truncate or overflow; tensors must be `float32` and contiguous (`x` 3-D indexing assumes a contiguous `(pop, batch, in)` layout).
- The fallback allocates `out`/noise on `weight.device` / `seeds.device` respectively, so `seeds`, `x`, `weight` must all live on the same device.
- `compute_counter_offsets` advances a ≥2-D block by `_ceil_div4(a+b)` counters, but `generate_lora_factors` actually starts `B` at `counter + _ceil_div4(a)` and lets it run `ceil(b/4)` calls (`a = rank*k`, `b = d*rank`), so it can occupy more counters than were budgeted — e.g. `a = b = 1` occupies counters 0 and 4 while the next block also starts at 4, so adjacent blocks can alias for tiny layers (`a ≤ 2` and `b ≤ 2`).
