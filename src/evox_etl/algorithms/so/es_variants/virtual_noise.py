"""Deterministic virtual-population Gaussian noise for the virtual-ES family.

ETL is stateless-key based (no jump-ahead counter streams), so torch's
``triton_kernels.kernels.{philox,virtual_noise}`` is replaced here by a
deterministic **splitmix64 integer hash + Box-Muller** transform built purely
from ETL ops.  Given per-individual integer ``seeds`` and a flat element
``offset``, :func:`virtual_normal` regenerates the SAME ``(pop_size, n_elements)``
standard-normal noise on every call, so the problem's forward pass and the
algorithm's gradient estimate stay consistent WITHOUT ever materialising a full
``(pop_size, dim)`` population (the algorithm stores only ``center`` ``(dim,)``
+ ``seeds`` ``(pop_size,)``).

This is the SHARED contract between ``virtual_es``/``virtual_lora_es``
(algorithms, which compute the gradient) and
``evox_etl.problems.neuroevolution.virtual_problem`` (which runs the forward
pass).  numpy is used host-side only (static shape arithmetic).

Noise indexing scheme (mirrors torch ``virtual_noise.py``): for a block whose
flat start offset is ``off`` and shape ``(out, in)``, weight element ``(j, k)``
uses element index ``off + j * in + k``; a bias block ``(out,)`` immediately
after uses ``off + out * in + j``.
"""

from __future__ import annotations

import math
from typing import Sequence

import etl
import etl.numpy as enp
import numpy as np

__all__ = [
    "compute_offsets",
    "compute_counter_offsets",
    "virtual_normal",
    "lora_factors",
]

#: Golden-ratio constant of splitmix64, as a signed int64 (the unsigned value
#: has the high bit set; int64 multiply wraps mod 2**64).
_GAMMA_I64 = 0x9E3779B97F4A7C15 - (1 << 64)


def compute_offsets(param_shapes: Sequence[Sequence[int]]) -> list[int]:
    """Cumulative flat-element offsets (starting index) of each parameter block."""
    offsets: list[int] = []
    cur = 0
    for shape in param_shapes:
        offsets.append(cur)
        n = 1
        for s in shape:
            n *= s
        cur += n
    return offsets


def _ceil_div4(x: int) -> int:
    """Round ``x`` up to a multiple of 4 (the LoRA sub-stream stride)."""
    return ((x + 3) // 4) * 4


def compute_counter_offsets(param_shapes: Sequence[Sequence[int]], lora_rank: int) -> list[int]:
    """Non-overlapping counter offsets for the LoRA factor streams of each block.

    A 1-D block ``(n,)`` consumes ``n`` values; a ``(d, k)`` block consumes
    ``rank * k`` (factor A) + ``d * rank`` (factor B), each rounded up to a
    multiple of 4.
    """
    offsets: list[int] = []
    cur = 0
    for shape in param_shapes:
        offsets.append(cur)
        if len(shape) == 1:
            n = shape[0]
        else:
            d = 1
            for s in shape[:-1]:
                d *= s
            n = lora_rank * shape[-1] + d * lora_rank
        cur += _ceil_div4(n)
    return offsets


def _logical_rshift(x: etl.SymbolicTensor, shift: int) -> etl.SymbolicTensor:
    """Logical (zero-filling) right shift, robust to arithmetic ``>>`` semantics."""
    return etl.bitwise_and(etl.bitwise_right_shift(x, shift), (1 << (64 - shift)) - 1)


def _splitmix64(z: etl.SymbolicTensor) -> etl.SymbolicTensor:
    """The canonical splitmix64 finalizer, wrapping mod 2**64 (int64 ops)."""
    z = etl.bitwise_xor(z, _logical_rshift(z, 30))
    z = z * _GAMMA_I64
    z = etl.bitwise_xor(z, _logical_rshift(z, 27))
    z = z * _GAMMA_I64
    z = etl.bitwise_xor(z, _logical_rshift(z, 31))
    return z


def virtual_normal(seeds: etl.SymbolicTensor, offset: int, n_elements: int) -> etl.SymbolicTensor:
    """Deterministic ``(pop_size, n_elements)`` standard-normal noise (float32).

    :param seeds: ``(pop_size,)`` int tensor of per-individual seeds.
    :param offset: Static flat element offset of this block.
    :param n_elements: Static number of elements per individual.
    """
    seeds = etl.cast(enp.reshape(seeds, (-1,)), np.int64)
    flat = enp.arange(n_elements, dtype=np.int64) + offset
    z = etl.bitwise_xor(
        enp.expand_dims(seeds, axis=1),
        enp.expand_dims(flat, axis=0) * _GAMMA_I64,
    )
    z = _splitmix64(z)
    high32 = etl.bitwise_and(etl.bitwise_right_shift(z, 32), 0xFFFFFFFF)
    low32 = etl.bitwise_and(z, 0xFFFFFFFF)
    u1 = etl.cast(low32, np.float32) * (1.0 / float(1 << 32))
    u2 = etl.cast(high32, np.float32) * (1.0 / float(1 << 32))
    r = etl.sqrt(-2.0 * etl.log(etl.clamp(u1, 1e-10, 1.0)))
    return r * etl.cos(2.0 * math.pi * u2)


def lora_factors(
    seeds: etl.SymbolicTensor,
    shape: Sequence[int],
    rank: int,
    counter: int,
) -> etl.SymbolicTensor | tuple[etl.SymbolicTensor, etl.SymbolicTensor]:
    """Deterministic LoRA factors for a block, batched over individuals.

    - 1-D ``(n,)``: returns a flat ``(pop_size, n)`` noise tensor.
    - >=2-D ``(d, k)`` (``d = prod(shape[:-1])``): returns ``(A, B)`` with
      ``A`` ``(pop_size, rank, k)`` and ``B`` ``(pop_size, d, rank)`` so that
      ``B @ A`` is the per-individual low-rank delta.
    """
    pop = int(seeds.shape[0])
    if len(shape) == 1:
        return virtual_normal(seeds, counter, shape[0])
    d = 1
    for s in shape[:-1]:
        d *= s
    k = shape[-1]
    a_elems = rank * k
    a = enp.reshape(virtual_normal(seeds, counter, a_elems), (pop, rank, k))
    b = enp.reshape(
        virtual_normal(seeds, counter + _ceil_div4(a_elems), d * rank), (pop, d, rank)
    )
    return a, b
