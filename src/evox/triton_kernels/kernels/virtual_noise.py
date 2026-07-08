"""Virtual (never-materialized) noise fused into linear / matmul kernels.

This module implements *virtual* perturbation of neural-network weight matrices:
instead of generating a full ``(pop_size, out_features, in_features)`` noise
tensor (which would be prohibitively large), the noise is generated
on-the-fly inside the matmul kernel and added to the weight tile in registers.

The population-based zeroth-order / evolution-strategies model is:

    Y[i] = X[i] @ (W + sigma * N_i)^T + (b + sigma * nb_i)

where ``N_i`` is a full ``(out_features, in_features)`` noise matrix
that is *unique* per individual ``i`` (derived from ``seed_i``). The corresponding
gradient estimate w.r.t. ``W`` is:

    grad_W[j, k] = sum_i fitness_i * N_i[j, k] / (pop_size * sigma)

**CRITICAL requirement**: the forward pass (:func:`virtual_perturbed_linear`) and
the gradient estimate (:func:`virtual_weight_gradient` / :func:`virtual_bias_gradient`)
must regenerate the *exact same* noise ``N_i`` for a given ``(seed_i, offset,
element_index)`` triple. Each path (Triton / PyTorch) is internally self-consistent;
the two paths do not need to match each other.

### RNG: fast centered-uniform approximation

This is a forward-only performance demo. The noise is generated with a **cheap
centered-uniform RNG** (uniform ``[0, 1)`` shifted by ``-0.5``, i.e. values in
``[-0.5, 0.5)`` with mean ~0 and variance 1/12) rather than a standard normal.
This deliberately avoids the transcendental functions (``sqrt`` / ``log`` /
``cos`` of Box-Muller, or the Philox path behind ``tl.randn``) which are the
expensive part of the noise generation: a centered uniform is just
``tl.rand(seed, offsets) - 0.5`` and fuses trivially with no overhead. RNG
*distributional quality* is irrelevant for this demo — only determinism (the
same ``(seed, element_index)`` must always yield the same value, so the forward
and gradient paths regenerate identical noise) matters, and that is preserved.

### Noise indexing scheme

For a parameter block at flat element offset ``off``:

- Weight element ``(j, k)`` of shape ``(out, in)``:
  ``noise = PRNG(seed_i, off + j * in + k)``
- Bias element ``j`` of shape ``(out,)`` (placed immediately after the weight
  block, so the caller passes ``offset = off + out * in``):
  ``noise = PRNG(seed_i, offset + j)``

``offset`` is the cumulative element count across all preceding parameter blocks
(see :func:`compute_offsets`).
"""

from typing import Optional

import torch

from ..backend import has_triton
from ..op_register import register_triton_op

if has_triton():
    import triton
    import triton.language as tl


# ---------------------------------------------------------------------------
# CPU (PyTorch fallback) deterministic PRNG — splitmix64 based.
#
# This is the SINGLE source of truth for noise on the fallback path. Both
# :func:`virtual_perturbed_linear` (forward) and
# :func:`virtual_weight_gradient` / :func:`virtual_bias_gradient` (gradient)
# route their weight/bias noise through :func:`_cpu_normal_noise` so that the
# exact same noise is regenerated for the same ``(seeds, offset, layout)``.
# ---------------------------------------------------------------------------

#: Golden-ratio constant for splitmix64 (unsigned 64-bit representation).
_SPLITMIX64_GAMMA = 0x9E3779B97F4A7C15
#: Signed int64 representation of the gamma constant (high bit is set, so the
#: unsigned value exceeds int64 max). PyTorch int64 multiply/add/xor wrap mod
#: 2**64 just like C unsigned arithmetic.
_SPLITMIX64_GAMMA_I64 = _SPLITMIX64_GAMMA - (1 << 64)
#: 2**64 mask (for logical right-shift reconstruction).
_MASK64 = (1 << 64) - 1


def _cpu_logical_rshift(x: torch.Tensor, shift: int) -> torch.Tensor:
    """Logical (zero-filling) right shift on (possibly negative) int64 tensors.

    PyTorch's ``>>`` on signed int64 is *arithmetic* (sign-extending). For
    values whose high bit is set (i.e. that represent large unsigned 64-bit
    integers), arithmetic shift corrupts the bits. We reconstruct the logical
    shift by masking off the sign-extended bits.

    :param x: int64 tensor.
    :param shift: Number of bits to shift right (0 <= shift < 64).
    :return: Logical right-shifted int64 tensor.
    """
    return (x >> shift) & ((1 << (64 - shift)) - 1)


def _splitmix64_step(z: torch.Tensor) -> torch.Tensor:
    """One round of the splitmix64 mixing function (operates in-place logically).

    Given a 64-bit state ``z``, returns ``z'`` such that the full 64-bit result
    matches the canonical splitmix64 finalizer:

        z' = (z ^ (z >> 30)) * GAMMA
        z' = (z' ^ (z' >> 27)) * GAMMA
        z' = z' ^ (z' >> 31)

    All arithmetic wraps mod 2**64. PyTorch int64 multiply/add/xor wrap
    correctly; logical right shifts use :func:`_cpu_logical_rshift`.

    :param z: int64 tensor of states.
    :return: int64 tensor of mixed (64-bit) values.
    """
    z = z ^ _cpu_logical_rshift(z, 30)
    z = z * _SPLITMIX64_GAMMA_I64
    z = z ^ _cpu_logical_rshift(z, 27)
    z = z * _SPLITMIX64_GAMMA_I64
    z = z ^ _cpu_logical_rshift(z, 31)
    return z


def _cpu_normal_noise(seeds: torch.Tensor, n_elements: int, offset: int) -> torch.Tensor:
    """Generate ``(pop_size, n_elements)`` *cheap* centered-uniform noise deterministically.

    This is NOT a standard-normal distribution. It is a deliberately cheap
    centered-uniform approximation: a single uniform value in ``[0, 1)`` (derived
    from the low 32 bits of a splitmix64 hash) shifted by ``-0.5`` so the output
    lies in ``[-0.5, 0.5)`` with mean ~0 and variance 1/12.

    This is a **forward-only performance demo**: RNG *distributional quality* is
    irrelevant here — only *determinism* matters (the forward and gradient paths
    must regenerate identical noise for the same ``(seed, element_index)``), and
    that is fully preserved. Avoiding the transcendental functions of Box-Muller
    (``sqrt`` / ``log`` / ``cos``) keeps generation as cheap as possible. The
    output is trivially finite (bounded in ``[-0.5, 0.5)``), so it can never
    produce ``inf`` / ``nan``.

    The element index is ``offset + flat_index`` where ``flat_index`` ranges
    over ``[0, n_elements)``. For a weight block of shape ``(out, in)`` the
    flat index is ``j * in + k`` (row-major); for a bias block ``(out,)`` it is
    just ``j``. Forward and gradient MUST call this helper with the same
    ``offset`` and the same flat layout for the noise to match exactly.

    :param seeds: 1-D int64/int32 tensor of per-individual seeds, shape
        ``(pop_size,)``.
    :param n_elements: Number of output values per individual.
    :param offset: Flat element offset for this block's noise.
    :return: ``(pop_size, n_elements)`` float32 centered-uniform noise tensor
        with values in ``[-0.5, 0.5)``.
    """
    seeds = seeds.to(torch.int64).reshape(-1)
    # Flat element indices: offset + [0, n_elements)
    flat_idx = torch.arange(n_elements, device=seeds.device, dtype=torch.int64) + offset
    # Combine seed and element index into a unique 64-bit hash input.
    # ``seed * 2**? ``  could collide; use seed ^ (index * GAMMA) instead so that
    # distinct (seed, index) pairs hash distinctly even for adjacent seeds.
    z = seeds.unsqueeze(1) ^ (flat_idx.unsqueeze(0) * _SPLITMIX64_GAMMA_I64)
    z = _splitmix64_step(z)
    # Extract one uniform in [0, 1) from the low 32 bits, then center it to
    # [-0.5, 0.5). No transcendentals — deterministic & cheap.
    low32 = z & 0xFFFFFFFF
    u = low32.to(torch.float32) * (1.0 / float(1 << 32))
    return u - 0.5


# ---------------------------------------------------------------------------
# compute_offsets — pure Python helper (not a registered op).
# ---------------------------------------------------------------------------


def compute_offsets(param_shapes: list[tuple]) -> list[int]:
    """Compute cumulative flat-element offsets for a list of parameter blocks.

    For each block the number of elements is ``prod(shape)``. The returned list
    holds the *starting* offset of each block (the cumulative element count of
    all preceding blocks).

    Example::

        >>> compute_offsets([(256, 784), (256,), (10, 256), (10,)])
        [0, 200704, 200960, 203520]

    :param param_shapes: List of parameter block shapes (each a tuple of ints).
    :return: List of starting offsets (one per block), length
        ``len(param_shapes)``.
    """
    offsets = []
    cur = 0
    for shape in param_shapes:
        offsets.append(cur)
        n_elements = 1
        for s in shape:
            n_elements *= s
        cur += n_elements
    return offsets


# ---------------------------------------------------------------------------
# Fake (abstract evaluation) functions for torch.compile tracing.
# ---------------------------------------------------------------------------


def _virtual_perturbed_linear_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias,
    seeds: torch.Tensor,
    sigma: float,
    offset: int,
) -> torch.Tensor:
    pop_size = seeds.shape[0]
    out_features = weight.shape[0]
    if x.dim() == 2:
        batch = x.shape[0]
    else:
        batch = x.shape[1]
    return torch.empty(pop_size, batch, out_features, dtype=x.dtype, device=x.device)


def _virtual_weight_gradient_fake(
    fitness: torch.Tensor,
    seeds: torch.Tensor,
    weight_shape: list[int],
    sigma: float,
    pop_size: int,
    offset: int,
) -> torch.Tensor:
    return torch.empty(*weight_shape, dtype=fitness.dtype, device=fitness.device)


def _virtual_bias_gradient_fake(
    fitness: torch.Tensor,
    seeds: torch.Tensor,
    bias_shape: list[int],
    sigma: float,
    pop_size: int,
    offset: int,
) -> torch.Tensor:
    return torch.empty(*bias_shape, dtype=fitness.dtype, device=fitness.device)


# ---------------------------------------------------------------------------
# Triton kernels + launchers (defined only when Triton is available).
#
# NOTE: the ``@triton.jit`` kernels are defined inside the ``if has_triton():``
# block (because ``triton`` is only imported there). The launcher functions
# (``_triton_*``) live at module level so they can be passed to
# ``register_triton_op`` regardless of Triton availability.
# ---------------------------------------------------------------------------


if has_triton():

    # ``_tl_noise_block`` MUST be a ``@triton.jit`` function because it is called
    # from inside the ``@triton.jit`` kernels below. A plain module-level Python
    # function reference would trigger a Triton ``NameError`` at compile time
    # ("Cannot access global variable ... from within @jit'ed function").
    #
    # Fast centered-uniform RNG: ``tl.rand(seed, offsets)`` produces deterministic
    # uniform [0, 1) values given ``(seed, offsets)`` and has NO transcendentals,
    # so it fuses trivially. Centering by 0.5 gives a mean-0, variance-1/12 value
    # in [-0.5, 0.5) — a cheap approximation used by this forward-only demo where
    # RNG distributional quality is irrelevant (only determinism matters). The
    # SAME helper is used by all three kernels so the forward and gradient paths
    # regenerate identical noise.
    @triton.jit
    def _tl_noise_block(seed, offsets):
        return tl.rand(seed, offsets) - 0.5

    @triton.jit
    def _virtual_perturbed_linear_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        seeds_ptr,
        out_ptr,
        batch,
        in_features,
        out_features,
        offset,
        bias_offset,
        sigma,
        has_bias: tl.constexpr,
        x_is_per_individual: tl.constexpr,
        BLOCK_OUT: tl.constexpr,
        BLOCK_IN: tl.constexpr,
        BLOCK_BATCH: tl.constexpr,
    ):
        """Fused virtual-noise perturbed linear kernel.

        Grid: ``(pop_size, n_out_tiles)``. Each program handles one individual
        and one output-feature tile (``BLOCK_OUT`` outputs). The weight tile
        ``W[out_tile, :]`` is loaded into registers, the corresponding noise
        tile is generated on-the-fly from the individual's seed, added to the
        weight in registers, and the matmul against the (broadcast or
        per-individual) input tile is accumulated via ``tl.dot``. The noise is
        never materialized in global memory.

        Weight element ``(j, k)`` noise index: ``offset + j * in + k``.
        Bias element ``j`` noise index: ``bias_offset + j``.
        """
        pid_pop = tl.program_id(axis=0)
        pid_out = tl.program_id(axis=1)
        seed = tl.load(seeds_ptr + pid_pop).to(tl.int32)

        out_start = pid_out * BLOCK_OUT
        out_offs = out_start + tl.arange(0, BLOCK_OUT)
        out_mask = out_offs < out_features

        # Bias contribution for this output tile. It is independent of the
        # inner (reduction) dimension, so it is computed once here. Bias
        # element ``j`` noise index: ``bias_offset + j``.
        if has_bias:
            b_tile = tl.load(bias_ptr + out_offs, mask=out_mask, other=0.0)
            b_noise_offs = bias_offset + out_offs
            nb_tile = _tl_noise_block(seed, b_noise_offs)
            b_pert = b_tile + sigma * nb_tile
        else:
            b_pert = tl.zeros([BLOCK_OUT], dtype=tl.float32)

        # Iterate over batch tiles and compute Y[b, out_tile] = X[b,:] @ w_pert^T.
        # The inner (reduction) dimension is tiled in BLOCK_IN chunks so each
        # ``tl.dot`` operates on a (BLOCK_BATCH, BLOCK_IN) x (BLOCK_IN, BLOCK_OUT)
        # tile that fits in shared memory; the full reduction over ``in_features``
        # is accumulated in ``acc``. The weight noise for element ``(j, k)`` (with
        # ``k = kstart + local_k``) is still indexed ``offset + j * in + k``, so
        # the regenerated noise is byte-identical to a single-tile load and to the
        # weight-gradient kernel (which tiles the inner dim the same way).
        for bstart in range(0, batch, BLOCK_BATCH):
            b_offs = bstart + tl.arange(0, BLOCK_BATCH)
            b_mask = b_offs < batch
            acc = tl.zeros([BLOCK_BATCH, BLOCK_OUT], dtype=tl.float32)
            for kstart in range(0, in_features, BLOCK_IN):
                in_offs = kstart + tl.arange(0, BLOCK_IN)
                in_mask = in_offs < in_features

                # Load weight tile (BLOCK_OUT, BLOCK_IN) for this inner slice.
                w_ptrs = weight_ptr + out_offs[:, None] * in_features + in_offs[None, :]
                w_tile = tl.load(w_ptrs, mask=out_mask[:, None] & in_mask[None, :], other=0.0)

                # Generate noise tile in registers (never stored). Weight element
                # (j, k) -> noise index offset + j*in + k.
                noise_offs = offset + out_offs[:, None] * in_features + in_offs[None, :]
                noise_tile = _tl_noise_block(seed, noise_offs)

                # Perturbed weight tile (kept in registers).
                w_pert = w_tile + sigma * noise_tile

                # Input tile (BLOCK_BATCH, BLOCK_IN) for this inner slice.
                if x_is_per_individual:
                    x_ptrs = x_ptr + pid_pop * batch * in_features + b_offs[:, None] * in_features + in_offs[None, :]
                else:
                    x_ptrs = x_ptr + b_offs[:, None] * in_features + in_offs[None, :]
                x_tile = tl.load(x_ptrs, mask=b_mask[:, None] & in_mask[None, :], other=0.0)
                # acc: (BLOCK_BATCH, BLOCK_OUT) += X_tile @ w_pert^T
                acc += tl.dot(x_tile, tl.trans(w_pert))

            # Add bias tile (broadcast over batch dim) and store.
            acc = acc + b_pert[None, :]
            out_ptrs = out_ptr + pid_pop * batch * out_features + b_offs[:, None] * out_features + out_offs[None, :]
            tl.store(out_ptrs, acc, mask=b_mask[:, None] & out_mask[None, :])

    @triton.jit
    def _virtual_weight_gradient_kernel(
        fitness_ptr,
        seeds_ptr,
        grad_ptr,
        in_features,
        out_features,
        offset,
        pop_size,
        BLOCK_OUT: tl.constexpr,
        BLOCK_IN: tl.constexpr,
    ):
        """Fused virtual weight-gradient kernel.

        Grid: ``(n_in_tiles, n_out_tiles)`` over the weight tensor. Each program
        loops over the population, regenerates ``N_i[j, k]`` for every element
        in its tile, and accumulates ``fitness_i * N_i[j, k]``.
        """
        pid_in = tl.program_id(axis=0)
        pid_out = tl.program_id(axis=1)

        out_offs = pid_out * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
        in_offs = pid_in * BLOCK_IN + tl.arange(0, BLOCK_IN)
        out_mask = out_offs < out_features
        in_mask = in_offs < in_features

        acc = tl.zeros([BLOCK_OUT, BLOCK_IN], dtype=tl.float32)
        for i in range(pop_size):
            seed = tl.load(seeds_ptr + i).to(tl.int32)
            f = tl.load(fitness_ptr + i)
            noise_offs = offset + out_offs[:, None] * in_features + in_offs[None, :]
            noise = _tl_noise_block(seed, noise_offs)
            acc += f * noise

        grad_ptrs = grad_ptr + out_offs[:, None] * in_features + in_offs[None, :]
        tl.store(grad_ptrs, acc, mask=out_mask[:, None] & in_mask[None, :])

    @triton.jit
    def _virtual_bias_gradient_kernel(
        fitness_ptr,
        seeds_ptr,
        grad_ptr,
        out_features,
        offset,
        pop_size,
        BLOCK_OUT: tl.constexpr,
    ):
        """Fused virtual bias-gradient kernel.

        Grid: ``(n_out_tiles,)`` over the bias vector. Each program loops over
        the population, regenerates ``nb_i[j]`` for every element in its tile,
        and accumulates ``fitness_i * nb_i[j]``.
        """
        pid_out = tl.program_id(axis=0)
        out_offs = pid_out * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
        out_mask = out_offs < out_features

        acc = tl.zeros([BLOCK_OUT], dtype=tl.float32)
        for i in range(pop_size):
            seed = tl.load(seeds_ptr + i).to(tl.int32)
            f = tl.load(fitness_ptr + i)
            noise_offs = offset + out_offs
            noise = _tl_noise_block(seed, noise_offs)
            acc += f * noise

        tl.store(grad_ptr + out_offs, acc, mask=out_mask)


def _choose_num_stages(device: torch.device, per_stage_bytes: int, max_stages: int = 4) -> int:
    """Pick a software-pipelining depth that fits the device's shared-memory budget.

    Triton's default ``num_stages`` (>= 3) can over-allocate shared memory for
    large ``tl.dot`` tiles (e.g. the worst-case ``(BLOCK_BATCH=128, BLOCK_IN=512,
    BLOCK_OUT=64)`` forward tile), triggering an ``OutOfResources`` crash on
    shared-memory-limited GPUs (e.g. sm_86 with a ~99 KB per-block opt-in limit).

    This defensively queries the device's per-block shared-memory budget and
    clamps the pipeline depth to what fits. The footprint estimate
    (``per_stage_bytes``) is provided by the caller and is intentionally
    pessimistic vs Triton's real (MMA-tiled) usage, so the resulting
    ``num_stages`` is conservative — which is safe (it only avoids OOM).

    :param device: The device the kernel will run on.
    :param per_stage_bytes: Estimated shared-memory bytes consumed per pipeline
        stage for the kernel's largest tile (pessimistic).
    :param max_stages: Upper bound on the returned pipeline depth.
    :return: An int in ``[1, max_stages]``.
    """
    # Defensive shared-memory budget query. Different PyTorch/CUDA versions and
    # device types expose the limit under different attribute names; some GPUs
    # (e.g. sm_86) don't report the opt-in limit at all. On ANY failure we fall
    # back to the conservative default opt-in (48 KB).
    budget = 48 * 1024
    try:
        if device is not None and device.type == "cuda":
            props = torch.cuda.get_device_properties(device)
            for attr in (
                "max_shared_mem_per_block",
                "shared_memory_per_block_optin",
                "shared_memory_per_block",
                "max_shared_mem_per_sm",
            ):
                val = getattr(props, attr, None)
                if isinstance(val, int) and val > 0:
                    budget = val
                    break
    except Exception:
        budget = 48 * 1024

    if per_stage_bytes <= 0:
        return max_stages
    max_fit = budget // per_stage_bytes
    return min(max_stages, max(1, max_fit))


def _triton_virtual_perturbed_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias,
    seeds: torch.Tensor,
    sigma: float,
    offset: int,
) -> torch.Tensor:
    """Launch the fused Triton virtual-noise perturbed linear kernel."""
    pop_size = seeds.shape[0]
    out_features = weight.shape[0]
    in_features = weight.shape[1]
    x_is_per_individual = x.dim() == 3
    batch = x.shape[1] if x_is_per_individual else x.shape[0]

    out = torch.empty((pop_size, batch, out_features), dtype=torch.float32, device=weight.device)

    # Tile sizes (powers of two for tl.dot alignment requirements).
    BLOCK_OUT = min(max(triton.next_power_of_2(out_features), 16), 64)
    # BLOCK_IN is the inner (reduction) tile size: the kernel now tiles the
    # reduction dimension in BLOCK_IN chunks (see the inner ``for kstart`` loop),
    # so it can be reduced for shared-memory safety without changing results.
    # Capping at 128 keeps each ``tl.dot`` tile comfortably within the per-block
    # opt-in shared-memory limit (~100 KB on sm_86) for all batch tiles.
    BLOCK_IN = min(max(triton.next_power_of_2(in_features), 16), 128)
    # Adaptive BLOCK_BATCH: shrink the batch tile when the inner tile is large so
    # the A-tile (x_tile) fits comfortably in shared memory (avoids
    # OutOfResources on shared-memory-limited GPUs).
    batch_cap = 64 if BLOCK_IN >= 128 else 128
    BLOCK_BATCH = min(max(triton.next_power_of_2(batch), 16), batch_cap)

    n_out_tiles = triton.cdiv(out_features, BLOCK_OUT)
    grid = (pop_size, n_out_tiles)

    has_bias = bias is not None
    bias_offset = offset + out_features * in_features if has_bias else 0

    # Adaptive software-pipelining depth. The forward tile's worst-case shared
    # memory (A-tile + B-tile + accumulator) with Triton's default >=3 stages
    # can exceed the per-block limit on shared-memory-constrained GPUs.
    # per_stage_bytes is a pessimistic estimate of one stage's footprint.
    per_stage_bytes = (BLOCK_BATCH * BLOCK_IN + BLOCK_OUT * BLOCK_IN + BLOCK_BATCH * BLOCK_OUT) * 4
    num_stages = _choose_num_stages(weight.device, per_stage_bytes, max_stages=4)

    _virtual_perturbed_linear_kernel[grid](
        x,
        weight,
        bias if has_bias else x,  # dummy ptr; masked by the has_bias constexpr
        seeds,
        out,
        batch,
        in_features,
        out_features,
        offset,
        bias_offset,
        float(sigma),
        has_bias,
        x_is_per_individual,
        BLOCK_OUT=BLOCK_OUT,
        BLOCK_IN=BLOCK_IN,
        BLOCK_BATCH=BLOCK_BATCH,
        num_stages=num_stages,
    )
    return out


def _triton_virtual_weight_gradient(
    fitness: torch.Tensor,
    seeds: torch.Tensor,
    weight_shape: list[int],
    sigma: float,
    pop_size: int,
    offset: int,
) -> torch.Tensor:
    """Launch the fused Triton virtual weight-gradient kernel."""
    out_features, in_features = weight_shape
    grad = torch.empty((out_features, in_features), dtype=torch.float32, device=fitness.device)

    BLOCK_OUT = min(max(triton.next_power_of_2(out_features), 16), 64)
    BLOCK_IN = min(max(triton.next_power_of_2(in_features), 16), 4096)
    n_out_tiles = triton.cdiv(out_features, BLOCK_OUT)
    n_in_tiles = triton.cdiv(in_features, BLOCK_IN)
    grid = (n_in_tiles, n_out_tiles)

    # Elementwise noise + reduction (no tl.dot): footprint is the single tile.
    per_stage_bytes = (BLOCK_OUT * BLOCK_IN) * 4
    num_stages = _choose_num_stages(fitness.device, per_stage_bytes, max_stages=4)

    _virtual_weight_gradient_kernel[grid](
        fitness,
        seeds,
        grad,
        in_features,
        out_features,
        offset,
        pop_size,
        BLOCK_OUT=BLOCK_OUT,
        BLOCK_IN=BLOCK_IN,
        num_stages=num_stages,
    )
    # Kernel accumulates sum_i fitness_i * noise_i; normalize by (pop * sigma).
    grad = grad / (pop_size * sigma)
    return grad


def _triton_virtual_bias_gradient(
    fitness: torch.Tensor,
    seeds: torch.Tensor,
    bias_shape: list[int],
    sigma: float,
    pop_size: int,
    offset: int,
) -> torch.Tensor:
    """Launch the fused Triton virtual bias-gradient kernel."""
    out_features = bias_shape[0]
    grad = torch.empty((out_features,), dtype=torch.float32, device=fitness.device)

    BLOCK_OUT = min(max(triton.next_power_of_2(out_features), 16), 1024)
    n_out_tiles = triton.cdiv(out_features, BLOCK_OUT)
    grid = (n_out_tiles,)

    # 1-D elementwise tile; tiny footprint.
    per_stage_bytes = BLOCK_OUT * 4
    num_stages = _choose_num_stages(fitness.device, per_stage_bytes, max_stages=4)

    _virtual_bias_gradient_kernel[grid](
        fitness,
        seeds,
        grad,
        out_features,
        offset,
        pop_size,
        BLOCK_OUT=BLOCK_OUT,
        num_stages=num_stages,
    )
    grad = grad / (pop_size * sigma)
    return grad


# ---------------------------------------------------------------------------
# Public registered ops (PyTorch fallback is the function body).
# ---------------------------------------------------------------------------


@register_triton_op(fake_fn=_virtual_perturbed_linear_fake, triton_fn=_triton_virtual_perturbed_linear)
def virtual_perturbed_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    seeds: torch.Tensor,
    sigma: float,
    offset: int,
) -> torch.Tensor:
    """Virtual-noise perturbed linear transformation.

    For each individual ``i`` computes::

        Y[i] = X[i] @ (W + sigma * N_i)^T + (b + sigma * nb_i)

    where ``N_i`` is a full ``(out_features, in_features)`` Gaussian noise
    matrix generated on-the-fly from ``seed_i`` and the block ``offset``. The
    noise is never materialized as a full tensor in the Triton path; on the
    CPU fallback it is generated per-individual (performance is not critical
    on CPU).

    Weight element ``(j, k)`` uses noise element index ``offset + j * in + k``;
    bias element ``j`` uses ``offset + out * in + j``.

    :param x: Input tensor, either ``(batch, in_features)`` (shared across all
        individuals) or ``(pop_size, batch, in_features)`` (per-individual).
    :param weight: Weight tensor of shape ``(out_features, in_features)``.
    :param bias: Bias tensor of shape ``(out_features,)`` or ``None``.
    :param seeds: 1-D int tensor of per-individual seeds, shape ``(pop_size,)``.
    :param sigma: Perturbation scale (Python float).
    :param offset: Flat element offset for this block's noise.
    :return: Output tensor of shape ``(pop_size, batch, out_features)``.
    """
    pop_size = seeds.shape[0]
    out_features, in_features = weight.shape
    x_is_per_individual = x.dim() == 3
    if x_is_per_individual:
        batch = x.shape[1]
    else:
        batch = x.shape[0]

    out = torch.empty((pop_size, batch, out_features), dtype=torch.float32, device=weight.device)

    # The full weight noise for individual i uses element index
    # offset + j * in + k (row-major (out, in) layout).
    w_noise = _cpu_normal_noise(seeds, out_features * in_features, offset)
    w_noise = w_noise.reshape(pop_size, out_features, in_features)

    if bias is not None:
        bias_offset = offset + out_features * in_features
        b_noise = _cpu_normal_noise(seeds, out_features, bias_offset)
    else:
        bias_offset = None
        b_noise = None

    for i in range(pop_size):
        w_pert = weight + sigma * w_noise[i]
        if x_is_per_individual:
            xi = x[i]
        else:
            xi = x
        yi = xi.to(torch.float32) @ w_pert.to(torch.float32).t()
        if bias is not None:
            b_pert = bias + sigma * b_noise[i]
            yi = yi + b_pert.to(torch.float32)
        out[i] = yi
    return out


@register_triton_op(fake_fn=_virtual_weight_gradient_fake, triton_fn=_triton_virtual_weight_gradient)
def virtual_weight_gradient(
    fitness: torch.Tensor,
    seeds: torch.Tensor,
    weight_shape: list[int],
    sigma: float,
    pop_size: int,
    offset: int,
) -> torch.Tensor:
    """Population-based virtual weight gradient estimate.

    Computes::

        grad[j, k] = sum_i fitness_i * N_i[j, k] / (pop_size * sigma)

    regenerating the *same* noise ``N_i`` as :func:`virtual_perturbed_linear`
    for weight element ``(j, k)`` (element index ``offset + j * in + k``).

    :param fitness: 1-D float tensor of per-individual fitness, ``(pop_size,)``.
    :param seeds: 1-D int tensor of per-individual seeds, ``(pop_size,)``.
    :param weight_shape: Target weight shape ``(out_features, in_features)``.
    :param sigma: Perturbation scale used in the forward pass.
    :param pop_size: Population size (number of individuals).
    :param offset: Flat element offset for this block's noise.
    :return: Gradient tensor of shape ``weight_shape``.
    """
    out_features, in_features = weight_shape
    w_noise = _cpu_normal_noise(seeds, out_features * in_features, offset)
    w_noise = w_noise.reshape(pop_size, out_features, in_features)
    grad = torch.einsum("i,ijk->jk", fitness.to(torch.float32), w_noise)
    grad = grad / (pop_size * sigma)
    return grad.reshape(weight_shape)


@register_triton_op(fake_fn=_virtual_bias_gradient_fake, triton_fn=_triton_virtual_bias_gradient)
def virtual_bias_gradient(
    fitness: torch.Tensor,
    seeds: torch.Tensor,
    bias_shape: list[int],
    sigma: float,
    pop_size: int,
    offset: int,
) -> torch.Tensor:
    """Population-based virtual bias gradient estimate.

    Computes::

        grad[j] = sum_i fitness_i * nb_i[j] / (pop_size * sigma)

    regenerating the *same* bias noise ``nb_i`` as the bias contribution of
    :func:`virtual_perturbed_linear` for bias element ``j`` (element index
    ``offset + j``).

    :param fitness: 1-D float tensor of per-individual fitness, ``(pop_size,)``.
    :param seeds: 1-D int tensor of per-individual seeds, ``(pop_size,)``.
    :param bias_shape: Target bias shape ``(out_features,)``.
    :param sigma: Perturbation scale used in the forward pass.
    :param pop_size: Population size (number of individuals).
    :param offset: Flat element offset for this block's bias noise (the caller
        should pass the offset pointing at the *bias region*, i.e.
        ``weight_offset + out_features * in_features``).
    :return: Gradient tensor of shape ``bias_shape``.
    """
    out_features = bias_shape[0]
    b_noise = _cpu_normal_noise(seeds, out_features, offset)
    grad = torch.einsum("i,ij->j", fitness.to(torch.float32), b_noise)
    grad = grad / (pop_size * sigma)
    return grad.reshape(bias_shape)
