"""Philox4x32-10 counter-based pseudo-random number generator (PRNG).

This module provides deterministic, seedable random number generation using the
Philox4x32-10 algorithm (the same family used by cuRAND / NumPy's Philox bit
generator). Two public ops are registered via
:func:`~evox.triton_kernels.register_triton_op`:

- :func:`philox_uniform` — uniform floats in ``[0, 1)`` (24 mantissa bits).
- :func:`philox_normal`  — standard normal floats via Box-Muller transform.

On CUDA devices with Triton installed the ops dispatch to hand-written Triton
kernels; on every other device (CPU, MPS, …) they fall back to the pure
PyTorch implementations. The two paths are designed to produce **identical**
output for the same inputs.

The algorithm follows the canonical Philox4x32-10 specification:

- Multiplication keys: ``M0 = 0xD2511F53``, ``M1 = 0xCD9E8D57``
- Key bump constants:  ``W0 = 0x9E3779B9``, ``W1 = 0xBB67AE85``
- 10 rounds of the Feistel-mixing ``single_round``.
- Each Philox *call* consumes one (4×uint32) counter and produces 4 uint32
  outputs. The low counter word ``c0`` is incremented per call so successive
  groups of 4 outputs are independent.
"""

import math

import torch

from ..backend import has_triton
from ..op_register import register_triton_op

if has_triton():
    import triton
    import triton.language as tl

#: Multiplication key 0 for Philox4x32-10 (static Python int).
PHILOX_M4x32_0 = 0xD2511F53
#: Multiplication key 1 for Philox4x32-10 (static Python int).
PHILOX_M4x32_1 = 0xCD9E8D57
#: Key-bump constant 0 for Philox4x32-10 (static Python int).
PHILOX_W32_0 = 0x9E3779B9
#: Key-bump constant 1 for Philox4x32-10 (static Python int).
PHILOX_W32_1 = 0xBB67AE85
#: 32-bit mask.
MASK = 0xFFFFFFFF

# Number of Philox rounds (the "10" in Philox4x32-10).
_ROUNDS = 10


def _philox_uniform_fallback(seeds: torch.Tensor, n: int, counter: int = 0) -> torch.Tensor:
    """Pure-PyTorch Philox4x32-10 uniform generator (the fallback path).

    Produces ``n`` uniform float32 values in ``[0, 1)`` per seed using 24
    mantissa bits: ``(uint32 >> 8) * (1 / 2**24)``.

    :param seeds: 1-D ``int64`` tensor of per-individual seeds.
    :param n: Number of output values per seed.
    :param counter: Starting 64-bit counter value (jump-ahead offset).
    :return: ``(pop_size, n)`` ``float32`` tensor of uniform values in ``[0, 1)``.
    """
    pop = seeds.shape[0]
    k0 = (seeds & MASK).to(torch.int32)
    k1 = ((seeds >> 32) & MASK).to(torch.int32)
    n_calls = (n + 3) // 4
    c0 = torch.full((pop, n_calls), counter & MASK, dtype=torch.int32)
    c1 = torch.full((pop, n_calls), (counter >> 32) & MASK, dtype=torch.int32)
    idx = torch.arange(n_calls, dtype=torch.int32)
    # Counter increments in c0 per Philox call.
    c0 = (c0 + idx) & MASK
    c2 = torch.zeros((pop, n_calls), dtype=torch.int32)
    c3 = torch.zeros((pop, n_calls), dtype=torch.int32)
    kk0 = k0.unsqueeze(1).expand(pop, n_calls).clone()
    kk1 = k1.unsqueeze(1).expand(pop, n_calls).clone()
    for _ in range(_ROUNDS):
        p0 = (c0.to(torch.int64) & MASK) * (PHILOX_M4x32_0 & MASK)
        hi0 = ((p0 >> 32) & MASK).to(torch.int32)
        lo0 = (p0 & MASK).to(torch.int32)
        p1 = (c2.to(torch.int64) & MASK) * (PHILOX_M4x32_1 & MASK)
        hi1 = ((p1 >> 32) & MASK).to(torch.int32)
        lo1 = (p1 & MASK).to(torch.int32)
        c0, c1, c2, c3 = (hi1 ^ c1 ^ kk0), lo1, (hi0 ^ c3 ^ kk1), lo0
        kk0 = (((kk0.to(torch.int64) & MASK) + PHILOX_W32_0) & MASK).to(torch.int32)
        kk1 = (((kk1.to(torch.int64) & MASK) + PHILOX_W32_1) & MASK).to(torch.int32)
    out = torch.stack([c0, c1, c2, c3], dim=2).reshape(pop, n_calls * 4)[:, :n]
    uni = (out.to(torch.int64) & MASK) >> 8
    uni = uni.to(torch.float32) * (1.0 / (1 << 24))
    return uni


def _philox_normal_fallback(seeds: torch.Tensor, n: int, counter: int = 0) -> torch.Tensor:
    """Pure-PyTorch standard normal generator via Box-Muller (the fallback path).

    Generates ``ceil(n/2)`` uniform pairs ``(u1, u2)`` and applies the Box-Muller
    transform. If ``n`` is odd, the last (extra) value is discarded.

    :param seeds: 1-D ``int64`` tensor of per-individual seeds.
    :param n: Number of output values per seed.
    :param counter: Starting 64-bit counter value (jump-ahead offset).
    :return: ``(pop_size, n)`` ``float32`` tensor of standard normal values.
    """
    n_pairs = (n + 1) // 2
    u = _philox_uniform_fallback(seeds, 2 * n_pairs, counter)
    u = u.reshape(seeds.shape[0], n_pairs, 2)
    u1 = u[..., 0]
    u2 = u[..., 1]
    r = torch.sqrt(-2.0 * torch.log(torch.clamp(u1, min=1e-10)))
    z1 = r * torch.cos(2.0 * math.pi * u2)
    z2 = r * torch.sin(2.0 * math.pi * u2)
    out = torch.stack([z1, z2], dim=-1).reshape(seeds.shape[0], 2 * n_pairs)
    return out[:, :n]


# ---------------------------------------------------------------------------
# Fake (abstract evaluation) functions for torch.compile tracing.
# ---------------------------------------------------------------------------


def _philox_uniform_fake(seeds: torch.Tensor, n: int, counter: int = 0) -> torch.Tensor:
    return torch.empty(seeds.shape[0], n, device=seeds.device, dtype=torch.float32)


def _philox_normal_fake(seeds: torch.Tensor, n: int, counter: int = 0) -> torch.Tensor:
    return torch.empty(seeds.shape[0], n, device=seeds.device, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Triton kernel + launchers.
# ---------------------------------------------------------------------------

if has_triton():

    @triton.jit
    def _philox_uniform_kernel(
        out_ptr,
        seeds_ptr,
        n,
        counter_low,
        counter_high,
        pop_size,
        n_calls,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Triton kernel: Philox4x32-10 uniform ``[0,1)`` generation.

        Grid: ``(pop_size, n_call_tiles)`` where ``n_call_tiles = cdiv(n_calls, BLOCK_SIZE)``.
        Each program computes ``BLOCK_SIZE`` Philox *calls* (4 outputs each) for one
        individual, producing up to ``4*BLOCK_SIZE`` uniform floats, then stores the
        first ``n`` of them.
        """
        pid_pop = tl.program_id(axis=0)
        pid_calls = tl.program_id(axis=1)

        # Derive the key from the seed for this individual.
        seed = tl.load(seeds_ptr + pid_pop).to(tl.int64)
        k0 = ((seed & 0xFFFFFFFF) & 0xFFFFFFFF).to(tl.int32)
        k1 = (((seed >> 32) & 0xFFFFFFFF) & 0xFFFFFFFF).to(tl.int32)

        call_offsets = pid_calls * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        call_mask = call_offsets < n_calls

        # Counter words. c0 increments per call; c1 is the high counter word.
        c0 = ((counter_low & 0xFFFFFFFF) + call_offsets) & 0xFFFFFFFF
        c1 = tl.zeros([BLOCK_SIZE], dtype=tl.int32) + (counter_high & 0xFFFFFFFF)
        c2 = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
        c3 = tl.zeros([BLOCK_SIZE], dtype=tl.int32)

        kk0 = tl.zeros([BLOCK_SIZE], dtype=tl.int32) + k0
        kk1 = tl.zeros([BLOCK_SIZE], dtype=tl.int32) + k1

        mask32 = 0xFFFFFFFF

        for _ in tl.static_range(_ROUNDS):
            p0 = (c0.to(tl.int64) & mask32) * (PHILOX_M4x32_0 & mask32)
            hi0 = ((p0 >> 32) & mask32).to(tl.int32)
            lo0 = (p0 & mask32).to(tl.int32)
            p1 = (c2.to(tl.int64) & mask32) * (PHILOX_M4x32_1 & mask32)
            hi1 = ((p1 >> 32) & mask32).to(tl.int32)
            lo1 = (p1 & mask32).to(tl.int32)
            new_c0 = hi1 ^ c1 ^ kk0
            new_c1 = lo1
            new_c2 = hi0 ^ c3 ^ kk1
            new_c3 = lo0
            c0 = new_c0
            c1 = new_c1
            c2 = new_c2
            c3 = new_c3
            kk0 = (((kk0.to(tl.int64) & mask32) + PHILOX_W32_0) & mask32).to(tl.int32)
            kk1 = (((kk1.to(tl.int64) & mask32) + PHILOX_W32_1) & mask32).to(tl.int32)

        # Interleave the 4 uint32 outputs into BLOCK_SIZE*4 consecutive values.
        # Each call produces (c0, c1, c2, c3) at positions [4*j, 4*j+1, 4*j+2, 4*j+3].
        # We build the full flattened buffer then store only the valid [0, n) slice.
        # Convert uint32 -> float in [0,1): (u >> 8) * (1/2^24).
        scale = 1.0 / float(1 << 24)
        f0 = ((c0.to(tl.int64) & mask32) >> 8).to(tl.float32) * scale
        f1 = ((c1.to(tl.int64) & mask32) >> 8).to(tl.float32) * scale
        f2 = ((c2.to(tl.int64) & mask32) >> 8).to(tl.float32) * scale
        f3 = ((c3.to(tl.int64) & mask32) >> 8).to(tl.float32) * scale

        # Output offset for this (individual, call-tile): base = pid_pop * n
        out_base = pid_pop * n
        # Position of each call's 4 outputs.
        pos0 = call_offsets * 4 + 0
        pos1 = call_offsets * 4 + 1
        pos2 = call_offsets * 4 + 2
        pos3 = call_offsets * 4 + 3
        m0 = call_mask & (pos0 < n)
        m1 = call_mask & (pos1 < n)
        m2 = call_mask & (pos2 < n)
        m3 = call_mask & (pos3 < n)
        tl.store(out_ptr + out_base + pos0, f0, mask=m0)
        tl.store(out_ptr + out_base + pos1, f1, mask=m1)
        tl.store(out_ptr + out_base + pos2, f2, mask=m2)
        tl.store(out_ptr + out_base + pos3, f3, mask=m3)


def _triton_philox_uniform(seeds: torch.Tensor, n: int, counter: int = 0) -> torch.Tensor:
    """Launch the Triton Philox uniform kernel.

    Produces the same values as :func:`_philox_uniform_fallback` for identical
    inputs.
    """
    pop_size = seeds.shape[0]
    out = torch.empty((pop_size, n), device=seeds.device, dtype=torch.float32)
    n_calls = (n + 3) // 4

    BLOCK_SIZE = 256
    n_call_tiles = triton.cdiv(n_calls, BLOCK_SIZE)
    grid = (pop_size, n_call_tiles)

    counter_low = counter & MASK
    counter_high = (counter >> 32) & MASK

    _philox_uniform_kernel[grid](
        out,
        seeds,
        n,
        counter_low,
        counter_high,
        pop_size,
        n_calls,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def _triton_philox_normal(seeds: torch.Tensor, n: int, counter: int = 0) -> torch.Tensor:
    """Launch the Triton Philox normal kernel via Box-Muller.

    Generates ``2*ceil(n/2)`` uniforms using the uniform kernel, then applies
    Box-Muller. This guarantees bit-identical results with
    :func:`_philox_normal_fallback` because it reuses the exact same uniform
    stream and the same Box-Muller arithmetic.
    """
    n_pairs = (n + 1) // 2
    u = _triton_philox_uniform(seeds, 2 * n_pairs, counter)
    u = u.reshape(seeds.shape[0], n_pairs, 2)
    u1 = u[..., 0]
    u2 = u[..., 1]
    r = torch.sqrt(-2.0 * torch.log(torch.clamp(u1, min=1e-10)))
    z1 = r * torch.cos(2.0 * math.pi * u2)
    z2 = r * torch.sin(2.0 * math.pi * u2)
    out = torch.stack([z1, z2], dim=-1).reshape(seeds.shape[0], 2 * n_pairs)
    return out[:, :n]


# ---------------------------------------------------------------------------
# Public registered ops.
# ---------------------------------------------------------------------------


@register_triton_op(fake_fn=_philox_uniform_fake, triton_fn=_triton_philox_uniform)
def philox_uniform(seeds: torch.Tensor, n: int, counter: int = 0) -> torch.Tensor:
    """Generate deterministic uniform random floats in ``[0, 1)`` via Philox4x32-10.

    Each seed in ``seeds`` produces an independent stream of ``n`` values. The
    optional ``counter`` provides jump-ahead so multiple sub-streams (e.g. for
    different parameter blocks) can be drawn from the same seed without overlap.

    On CUDA devices with Triton, a fused Triton kernel is used; otherwise the
    pure-PyTorch fallback runs on all devices (CPU, MPS, …). Both paths produce
    identical output for the same inputs.

    :param seeds: 1-D ``int64`` tensor of per-individual seeds.
    :param n: Number of output values per seed.
    :param counter: Starting 64-bit counter value (jump-ahead offset). Default 0.
    :return: ``(pop_size, n)`` ``float32`` tensor of uniform values in ``[0, 1)``.
    """
    return _philox_uniform_fallback(seeds, n, counter)


@register_triton_op(fake_fn=_philox_normal_fake, triton_fn=_triton_philox_normal)
def philox_normal(seeds: torch.Tensor, n: int, counter: int = 0) -> torch.Tensor:
    """Generate deterministic standard normal floats via Philox4x32-10 + Box-Muller.

    Each seed in ``seeds`` produces an independent stream of ``n`` standard
    normal (mean 0, std 1) values using the Box-Muller transform applied to the
    Philox uniform stream.

    On CUDA devices with Triton, the uniform stream is generated by the Triton
    kernel and Box-Muller is applied in PyTorch; otherwise the pure-PyTorch
    fallback runs on all devices. Both paths produce identical output.

    :param seeds: 1-D ``int64`` tensor of per-individual seeds.
    :param n: Number of output values per seed.
    :param counter: Starting 64-bit counter value (jump-ahead offset). Default 0.
    :return: ``(pop_size, n)`` ``float32`` tensor of standard normal values.
    """
    return _philox_normal_fallback(seeds, n, counter)
