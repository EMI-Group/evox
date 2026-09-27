"""Functional ETL port of the torch evox VirtualLoRAES algorithm (step protocol).

A DISTINCT low-rank variant of :mod:`evox_etl.algorithms.so.es_variants.virtual_es`:
for a >= 2-D weight ``(d, k)`` the per-individual perturbation is not a full
Gaussian matrix but the low-rank product ``delta = B @ A`` with ``A`` of shape
``(rank, k)`` and ``B`` of shape ``(d, rank)``, so the search space shrinks from
``d * k`` to ``rank * (d + k)``.  1-D blocks keep full Gaussian noise.  The
factors are regenerated deterministically from the same seeds by
:mod:`evox_etl.algorithms.so.es_variants.virtual_noise`, so the problem's
forward pass and the algorithm's gradient estimate stay consistent while the
algorithm still stores only ``center`` + ``seeds`` (O(dim) memory).

``step`` mirrors the torch original in
``src/evox/algorithms/so/es_variants/virtual_lora_es.py`` with an extra
``lora_rank`` hyperparameter.  Plain (non-``@etl.defn``) functions: they may
only be called inside an active trace, since ETL has no eager mode.
"""

from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import numpy as np

import etl
import etl.numpy as enp

from ..._config_utils import (
    ArrayLike,
    bake_float32_constant,
    require_choice,
    require_gt,
    to_float_tuple,
)
from ._virtual_common import draw_seeds, normalize_param_shapes, param_dim, update_center
from .virtual_noise import compute_counter_offsets, lora_factors

Tensor = etl.SymbolicTensor

#: float32 dtype spoken by every state leaf.
F32 = np.dtype("float32")


@dataclass(frozen=True)
class VirtualLoRAESConfig:
    """Frozen hyperparameters (torch ``VirtualLoRAES.__init__`` minus ``device``).

    ``param_shapes`` is a tuple of int tuples and ``center_init`` a flat float32
    tuple; array-like input is normalized by :func:`make_virtual_lora_es`.
    ``dim`` is an evox_etl addition holding the eager total parameter count, so
    the workflow's ``_discover_pop_size``/monitor completion can read
    ``cfg.dim`` (``init``/``step`` recompute it from ``param_shapes``).
    """

    param_shapes: tuple[tuple[int, ...], ...]
    lora_rank: int
    pop_size: int
    center_init: tuple[float, ...]
    learning_rate: float
    noise_stdev: float
    optimizer: Literal["adam"] | None = None
    dim: int | None = None


def make_virtual_lora_es(
    param_shapes: Sequence[Sequence[int]],
    lora_rank: int,
    pop_size: int,
    center_init: ArrayLike,
    learning_rate: float,
    noise_stdev: float,
    optimizer: Literal["adam"] | None = None,
) -> VirtualLoRAESConfig:
    """Construct a :class:`VirtualLoRAESConfig`, normalizing and validating inputs.

    ``param_shapes`` becomes a tuple of int tuples, ``center_init`` a flat
    float32 tuple, and ``dim`` is filled eagerly with the total parameter count.
    Invalid hyperparameters raise ValueError.
    """
    require_gt("noise_stdev", noise_stdev, 0)
    require_gt("learning_rate", learning_rate, 0)
    require_gt("pop_size", pop_size, 0)
    require_gt("lora_rank", lora_rank, 0)
    require_choice("optimizer", optimizer, (None, "adam"))
    shapes = normalize_param_shapes(param_shapes)
    dim = param_dim(shapes)
    center = to_float_tuple(center_init, dtype=np.float32)
    if len(center) != dim:
        raise ValueError(
            "center_init length must equal the total number of parameters "
            f"(sum of element counts over param_shapes): expected {dim}, got {len(center)}"
        )
    return VirtualLoRAESConfig(
        param_shapes=shapes,
        lora_rank=lora_rank,
        pop_size=pop_size,
        center_init=center,
        learning_rate=learning_rate,
        noise_stdev=noise_stdev,
        optimizer=optimizer,
        dim=dim,
    )


@dataclass(frozen=True)
class VirtualLoRAESState:
    """VirtualLoRAES state: tensor leaves ONLY (``key`` last)."""

    center: Tensor  # (dim,) current search center
    seeds: Tensor  # (pop_size,) int64 seeds of the last generation
    exp_avg: Tensor  # (dim,) Adam first moment (always carried)
    exp_avg_sq: Tensor  # (dim,) Adam second moment (always carried)
    best_fitness: Tensor  # () best fitness seen so far
    key: Tensor  # () RNG key


def init(config: VirtualLoRAESConfig, key: Tensor) -> VirtualLoRAESState:
    """Build the initial VirtualLoRAES state (center from the config, seeded RNG)."""
    dim = param_dim(config.param_shapes)
    key, seeds = draw_seeds(key, config.pop_size)
    return VirtualLoRAESState(
        center=bake_float32_constant(config.center_init),
        seeds=seeds,
        exp_avg=enp.zeros((dim,), dtype=F32),
        exp_avg_sq=enp.zeros((dim,), dtype=F32),
        best_fitness=enp.full((), np.inf, dtype=F32),
        key=key,
    )


def step(
    config: VirtualLoRAESConfig,
    state: VirtualLoRAESState,
    evaluate: Callable[[tuple[Tensor, Tensor, float]], Tensor],
) -> VirtualLoRAESState:
    """Run ONE full generation: resample the seeds, evaluate the
    ``(center, seeds, sigma)`` payload, rebuild the matching LoRA factors and
    update the center from the fitness-weighted low-rank gradient estimate.

    ``evaluate`` is the workflow-injected traced closure (opaque; minimization
    semantics).
    """
    pop_size = config.pop_size
    rank = config.lora_rank
    sigma = config.noise_stdev
    key, seeds = draw_seeds(state.key, pop_size)

    fitness = evaluate((state.center, seeds, sigma))
    weighted_fitness = enp.expand_dims(fitness, axis=1)

    parts = []
    offsets = compute_counter_offsets(config.param_shapes, rank)
    for shape, counter in zip(config.param_shapes, offsets):
        factors = lora_factors(seeds, shape, rank, counter)
        if isinstance(factors, tuple):
            # >=2-D block: per-individual delta = B @ A has shape (d, k).
            a, b = factors  # (pop_size, rank, k) and (pop_size, d, rank)
            delta = etl.dot(b, a)  # (pop_size, d, k)
            weighted = delta * enp.expand_dims(weighted_fitness, axis=2)
            grad_part = etl.sum(weighted, axes=0) / (pop_size * sigma)  # (d, k)
            parts.append(enp.reshape(grad_part, (-1,)))
        else:
            # 1-D block: factors is the flat (pop_size, n) noise.
            grad_part = etl.sum(factors * weighted_fitness, axes=0)  # (n,)
            parts.append(grad_part / (pop_size * sigma))
    flat_grad = etl.concatenate(parts, axis=0)

    return update_center(config, state, flat_grad, fitness, seeds, key)
