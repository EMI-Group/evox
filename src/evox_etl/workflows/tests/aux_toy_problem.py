"""Minimal stateless toy problem for the auxiliary-history channel tests.

The workflow resolves a component's plain functions via
``importlib.import_module(type(config).__module__)`` (the functional
module convention), so every config used together by a workflow MUST live in
its OWN module. This file holds the toy problem config plus its ``evaluate``;
the toy algorithm configs live in ``aux_toy_algorithm.py`` /
``aux_toy_algorithm_no_hook.py``.

Stateless (no ``init``): the workflow hands it an ``EmptyState`` and
``evaluate`` threads it straight back. Fitness is the row-wise sphere function
on a ``(n, dim)`` batch, i.e. rank-1 fitness (single-objective).

No torch imports — numpy backend only.
"""

from __future__ import annotations

import dataclasses
from typing import Any

__all__ = ["AuxToySphere", "evaluate"]

import etl.numpy as enp


@dataclasses.dataclass(frozen=True)
class AuxToySphere:
    """Toy single-objective sphere problem (only the dimension is carried)."""

    dim: int = 3


def evaluate(config: AuxToySphere, state: Any, pop: Any) -> tuple[Any, Any]:
    """Evaluate a batch of points: row-wise sum of squares.

    :param pop: candidate batch, shape ``(n, dim)`` float32.
    :return: ``(fitness, problem_state)`` with fitness of shape ``(n,)`` float32.
    """
    fitness = enp.sum(pop * pop, axis=1)
    return fitness, state
