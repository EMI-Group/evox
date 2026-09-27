"""Shared helpers for the functional virtual-ES ports (VirtualES / VirtualLoRAES).

The two algorithms differ ONLY in how they turn the per-individual fitness into
a flat gradient estimate; the config-shape normalization, the key/seed drawing
and the center update (plain SGD or Adam) are identical.  Keeping them here
avoids duplicating the update rule between `virtual_es.py` and
`virtual_lora_es.py`.

numpy is used on the host side only (inside `make_*`); the functions called
inside a trace use etl ops exclusively.
"""

import math
from dataclasses import replace
from typing import Any, Literal, Protocol, Sequence

import etl
import etl.random as random

from .adam_step import adam_single_tensor

Tensor = etl.SymbolicTensor


class VirtualConfig(Protocol):
    """Structural contract of the two virtual-ES configs (duck-typed)."""

    learning_rate: float
    optimizer: Literal["adam"] | None


class VirtualState(Protocol):
    """Structural contract of the two virtual-ES states (duck-typed)."""

    center: Tensor  # (dim,)
    exp_avg: Tensor  # (dim,)
    exp_avg_sq: Tensor  # (dim,)
    best_fitness: Tensor  # ()


def normalize_param_shapes(
    param_shapes: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    """Normalize `param_shapes` to a tuple of int tuples, validating each shape.

    Raises ValueError when `param_shapes` is None or empty, an entry is empty,
    or any dimension is not a positive integer.
    """
    if param_shapes is None:
        raise ValueError("param_shapes must be a non-empty sequence of shapes, got None")
    shapes: list[tuple[int, ...]] = []
    for shape in param_shapes:
        dims = tuple(int(s) for s in shape)
        if not dims:
            raise ValueError(f"param_shapes entries must be non-empty, got {shape!r}")
        for s in dims:
            if s <= 0:
                raise ValueError(
                    f"param_shapes dimensions must be positive integers, got {dims!r}"
                )
        shapes.append(dims)
    if not shapes:
        raise ValueError("param_shapes must be a non-empty sequence of shapes")
    return tuple(shapes)


def param_dim(param_shapes: Sequence[Sequence[int]]) -> int:
    """Total flat element count of `param_shapes` (the length of `center`)."""
    return sum(math.prod(shape) for shape in param_shapes)


def draw_seeds(key: Tensor, pop_size: int) -> tuple[Tensor, Tensor]:
    """Split `key` and draw `(pop_size,)` int64 seeds in [0, 2**31).

    Returns `(advanced_key, seeds)`; the caller stores the advanced key in the
    state, mirroring the JAX-evoX key-threading convention and the torch
    reference's ``torch.randint(0, 2**31, (pop_size,), dtype=torch.int64)``.
    """
    key, subkey = random.split(key)
    seeds = random.randint(subkey, (pop_size,), 0, 2**31, dtype=etl.int64)
    return key, seeds


def update_center(
    config: VirtualConfig,
    state: VirtualState,
    flat_grad: Tensor,
    fitness: Tensor,
    seeds: Tensor,
    key: Tensor,
) -> Any:
    """Apply the family's shared center update and best-fitness bookkeeping.

    With `config.optimizer is None` the update is plain SGD (`center - lr * grad`)
    and the Adam moments are passed through UNCHANGED — they are only ever
    populated when `optimizer == "adam"`, so a plain-SGD run keeps them at their
    all-zero initialization.

    Returns a `dataclasses.replace`d copy of `state` with the new center, the
    freshly drawn seeds, the advanced key and the running best fitness.
    """
    if config.optimizer is None:
        center = state.center - config.learning_rate * flat_grad
        exp_avg, exp_avg_sq = state.exp_avg, state.exp_avg_sq
    else:
        center, exp_avg, exp_avg_sq = adam_single_tensor(
            state.center,
            flat_grad,
            state.exp_avg,
            state.exp_avg_sq,
            0.9,
            0.999,
            config.learning_rate,
        )
    best_fitness = etl.minimum(state.best_fitness, etl.min(fitness))
    return replace(
        state,
        center=center,
        seeds=seeds,
        exp_avg=exp_avg,
        exp_avg_sq=exp_avg_sq,
        best_fitness=best_fitness,
        key=key,
    )
