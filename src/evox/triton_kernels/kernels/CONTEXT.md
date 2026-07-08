# Triton Kernel Implementations

## Intent
Individual Triton kernel files, each demonstrating the full `register_triton_op` registration pattern: a `@triton.jit` kernel (defined inside `if has_triton():`), a pure-PyTorch `_fake` function for `torch.compile` tracing, a `_triton_*` launcher, and a public registered op whose body is the PyTorch (CPU) fallback. When Triton/CUDA is available the dispatcher routes CUDA tensors to the Triton kernel; otherwise the pure-PyTorch fallback runs on any device.

## API Surface
- `fused_add` — reference fused element-wise addition kernel (`fused_add.py`).
- `philox_uniform` / `philox_normal` — Philox4x32-10 deterministic PRNG (`philox.py`).
- LoRA noise helpers (`lora_noise.py`): `compute_counter_offsets`, `generate_lora_factors`, `lora_delta_output`, `lora_gradient`.
- Virtual noise helpers (`virtual_noise.py`):
  - `compute_offsets` — cumulative flat-element offsets for parameter blocks (pure Python).
  - `virtual_perturbed_linear` — fused virtual-noise perturbed linear / matmul forward pass.
  - `virtual_weight_gradient` / `virtual_bias_gradient` — regenerate the SAME noise as the forward pass for analytical gradient estimates.
  - `virtual_reduce_metric` — per-individual mean-absolute-value reduction `mean_k(|center[k] + sigma*noise[i,k]|)`, never materializing the noise tensor (one program per individual).
- All virtual-noise ops use a **fast centered-uniform RNG** (`tl.rand(seed, offsets) - 0.5` on the Triton path; splitmix64-hash uniform centered to `[-0.5, 0.5)` on the CPU fallback). This is a forward-only performance demo where RNG distributional quality is irrelevant — only determinism matters (forward ↔ gradient must regenerate identical noise for the same `(seed, element_index)` triple).

## Constraints
- **Triton is optional**: every `@triton.jit` kernel lives inside `if has_triton():`; the launcher functions and public ops are defined at module level so the module imports without Triton.
- **Noise consistency**: the forward op and its gradient ops MUST regenerate byte-identical noise for a given `(seed, offset, element_index)`. Use the shared `_tl_noise_block` (Triton) / `_cpu_normal_noise` (fallback) helpers for all kernels. The Triton path and the CPU fallback do NOT need to match each other — only each path must be internally self-consistent.
- **Pure PyTorch fallbacks**: fallback function bodies must be pure PyTorch (no NumPy) for GPU/compile compatibility.
- **Shared-memory safety**: the fused forward matmul kernel tiles its reduction dimension (`BLOCK_IN`) and selects an adaptive `num_stages` via `_choose_num_stages` to avoid `OutOfResources` on shared-memory-constrained GPUs (e.g. sm_86). Do NOT regress this.

## Routing Table
| Area | Path | Description |
|---|---|---|
| Fused add | `fused_add.py` | Reference fused addition kernel |
| Philox PRNG | `philox.py` | `philox_uniform` / `philox_normal` |
| LoRA noise | `lora_noise.py` | LoRA perturbation helpers |
| Virtual noise | `virtual_noise.py` | Virtual (never-materialized) noise fused into matmul + reduction ops |
