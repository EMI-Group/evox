"""Functional ETL port of the torch evox VirtualES algorithm (step protocol).

Instead of materialising a full ``(pop_size, dim)`` population, VirtualES
stores a center vector ``(dim,)`` plus ``(pop_size,)`` integer seeds; the
per-individual full-parameter Gaussian perturbations are regenerated
deterministically on demand by
:mod:`evox_etl.algorithms.so.es_variants.virtual_noise` (O(dim) memory instead
of O(pop_size * dim)).

``step`` mirrors the torch original in
``src/evox/algorithms/so/es_variants/virtual_es.py``: resample the seeds, call
``evaluate((center, seeds, sigma))`` (the workflow routes that payload to a
virtual-population problem), rebuild the SAME noise from those seeds, form the
fitness-weighted ES gradient estimate and update the center.

Plain (non-``@etl.defn``) functions: they may only be called inside an active
trace, since ETL has no eager mode.
"""

import math
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
from .virtual_noise import compute_offsets, virtual_normal

Tensor = etl.SymbolicTensor

#: float32 dtype spoken by every state leaf.
F32 = np.dtype("float32")


@dataclass(frozen=True)
class VirtualESConfig:
    """Frozen hyperparameters (torch ``VirtualES.__init__`` minus ``device``).

    ``param_shapes`` is a tuple of int tuples and ``center_init`` a flat float32
    tuple; array-like input is normalized by :func:`make_virtual_es`.  ``dim``
    is an evox_etl addition holding the eager total parameter count, so the
    workflow's ``_discover_pop_size``/monitor completion can read ``cfg.dim``
    (``init``/``step`` recompute it from ``param_shapes``).
    """

    param_shapes: tuple[tuple[int, ...], ...]
    pop_size: int
    center_init: tuple[float, ...]
    learning_rate: float
    noise_stdev: float
    optimizer: Literal["adam"] | None = None
    dim: int | None = None


def make_virtual_es(
    param_shapes: Sequence[Sequence[int]],
    pop_size: int,
    center_init: ArrayLike,
    learning_rate: float,
    noise_stdev: float,
    optimizer: Literal["adam"] | None = None,
) -> VirtualESConfig:
    """Construct a :class:`VirtualESConfig`, normalizing and validating inputs.

    ``param_shapes`` becomes a tuple of int tuples, ``center_init`` a flat
    float32 tuple, and ``dim`` is filled eagerly with the total parameter count.
    Invalid hyperparameters raise ValueError.
    """
    require_gt("noise_stdev", noise_stdev, 0)
    require_gt("learning_rate", learning_rate, 0)
    require_gt("pop_size", pop_size, 0)
    require_choice("optimizer", optimizer, (None, "adam"))
    shapes = normalize_param_shapes(param_shapes)
    dim = param_dim(shapes)
    center = to_float_tuple(center_init, dtype=np.float32)
    if len(center) != dim:
        raise ValueError(
            "center_init length must equal the total number of parameters "
            f"(sum of element counts over param_shapes): expected {dim}, got {len(center)}"
        )
    return VirtualESConfig(
        param_shapes=shapes,
        pop_size=pop_size,
        center_init=center,
        learning_rate=learning_rate,
        noise_stdev=noise_stdev,
        optimizer=optimizer,
        dim=dim,
    )


@dataclass(frozen=True)
class VirtualESState:
    """VirtualES state: tensor leaves ONLY (``key`` last)."""

    center: Tensor  # (dim,) current search center
    seeds: Tensor  # (pop_size,) int64 seeds of the last generation
    exp_avg: Tensor  # (dim,) Adam first moment (always carried)
    exp_avg_sq: Tensor  # (dim,) Adam second moment (always carried)
    best_fitness: Tensor  # () best fitness seen so far
    key: Tensor  # () RNG key


def init(config: VirtualESConfig, key: Tensor) -> VirtualESState:
    """Build the initial VirtualES state (center from the config, seeded RNG)."""
    dim = param_dim(config.param_shapes)
    key, seeds = draw_seeds(key, config.pop_size)
    return VirtualESState(
        center=bake_float32_constant(config.center_init),
        seeds=seeds,
        exp_avg=enp.zeros((dim,), dtype=F32),
        exp_avg_sq=enp.zeros((dim,), dtype=F32),
        best_fitness=enp.full((), np.inf, dtype=F32),
        key=key,
    )


def step(
    config: VirtualESConfig,
    state: VirtualESState,
    evaluate: Callable[[tuple[Tensor, Tensor, float]], Tensor],
) -> VirtualESState:
    """Run ONE full generation: resample the seeds, evaluate the
    ``(center, seeds, sigma)`` payload, rebuild the matching virtual noise and
    update the center from the fitness-weighted ES gradient estimate.

    ``evaluate`` is the workflow-injected traced closure (opaque; minimization
    semantics).
    """
    pop_size = config.pop_size
    sigma = config.noise_stdev
    key, seeds = draw_seeds(state.key, pop_size)

    fitness = evaluate((state.center, seeds, sigma))

    parts = []
    for shape, offset in zip(config.param_shapes, compute_offsets(config.param_shapes)):
        noise = virtual_normal(seeds, offset, math.prod(shape))  # (pop_size, n)
        grad_part = etl.sum(noise * enp.expand_dims(fitness, axis=1), axes=0)
        parts.append(grad_part / (pop_size * sigma))
    flat_grad = etl.concatenate(parts, axis=0)

    return update_center(config, state, flat_grad, fitness, seeds, key)


def monitor_candidate(payload: tuple[Tensor, Tensor, float]) -> Tensor:
    """Map the ``(center, seeds, sigma)`` evaluate payload onto the monitor candidate.

    The :class:`~evox_etl.workflows.EvalMonitorConfig` (SO path) concatenates the
    candidate with its ``(topk, dim)`` elite buffer, so it needs a ``(pop_size,
    dim)`` tensor. VirtualES never materialises that population — the only
    solution it owns is the ``(dim,)`` center — so the monitor is fed the center
    broadcast to ``(pop_size, dim)`` (every row is the same center, while the
    per-individual fitness still comes from the perturbed draws). This hook is
    only invoked when a monitor is configured, so the monitor-less path stays
    O(dim).
    """
    center, seeds, _ = payload
    return enp.broadcast_to(center, (int(seeds.shape[0]), int(center.shape[0])))


# Torch parity: `virtual_es.py:122` ends with the alias `VirtualLoRAES = VirtualES`.
# The evox_etl algorithm handle IS its config dataclass, so `VirtualES` here is
# the VirtualES config; the DISTINCT low-rank port lives in `virtual_lora_es.py`
# as `VirtualLoRAESConfig` (which the package `__init__`s expose as the
# torch-style bare name `VirtualLoRAES`).
VirtualES = VirtualESConfig
VirtualLoRAES = VirtualES