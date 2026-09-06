"""Functional port of the torch evox basic numerical problems (``basic.py``).

Each torch class becomes a frozen config dataclass mirroring the torch
``__init__`` signature (``shift``/``affine`` are numpy arrays; no ``device``).
``evaluate`` is a plain module-level function (no decorator — the workflow
traces it with ``etl.build``/``etl.trace``): the config's shift and affine
transform are applied to the input first (baked as graph constants), then the
true function is evaluated.  Calling the functions outside an active etl
trace raises ``etl.core.TraceError`` (expected).
"""

from __future__ import annotations

__all__ = [
    "ShiftAffineNumericalProblem",
    "Ackley",
    "Griewank",
    "Rastrigin",
    "Rosenbrock",
    "Schwefel",
    "Sphere",
    "Ellipsoid",
    "Zakharov",
    "Levy",
    "ackley_func",
    "griewank_func",
    "rastrigin_func",
    "rosenbrock_func",
    "schwefel_func",
    "sphere_func",
    "ellipsoid_func",
    "zakharov_func",
    "levy_func",
]

import dataclasses
import math
from typing import TYPE_CHECKING

import numpy as np

import etl
import etl.numpy as enp

if TYPE_CHECKING:
    from .state import ProblemState


def _validate_shift_affine(
    shift: np.ndarray | None, affine: np.ndarray | None
) -> None:
    """Static checks mirroring torch ``ShiftAffineNumericalProblem.__init__``."""
    if affine is not None:
        assert affine.ndim == 2 and affine.shape[0] == affine.shape[1], (
            "affine must be a square matrix"
        )
    if shift is not None:
        assert shift.ndim == 1, "shift must be a vector"
        if affine is not None:
            assert affine.shape[0] == shift.shape[0], (
                "affine and shift must have the same dimension"
            )


@dataclasses.dataclass(frozen=True)
class ShiftAffineNumericalProblem:
    """A numerical problem with a shift and affine transformations to the input points."""

    shift: np.ndarray | None = None
    affine: np.ndarray | None = None

    def __post_init__(self) -> None:
        _validate_shift_affine(self.shift, self.affine)


@dataclasses.dataclass(frozen=True)
class Ackley:
    """The Ackley function whose minimum is x = [0, ..., 0]"""

    a: float = 20.0
    b: float = 0.2
    c: float = 2 * math.pi
    shift: np.ndarray | None = None
    affine: np.ndarray | None = None

    def __post_init__(self) -> None:
        _validate_shift_affine(self.shift, self.affine)


@dataclasses.dataclass(frozen=True)
class Griewank:
    """The Griewank function whose minimum is x = [0, ..., 0]"""

    shift: np.ndarray | None = None
    affine: np.ndarray | None = None

    def __post_init__(self) -> None:
        _validate_shift_affine(self.shift, self.affine)


@dataclasses.dataclass(frozen=True)
class Rastrigin:
    """The Rastrigin function whose minimum is x = [0, ..., 0]"""

    shift: np.ndarray | None = None
    affine: np.ndarray | None = None

    def __post_init__(self) -> None:
        _validate_shift_affine(self.shift, self.affine)


@dataclasses.dataclass(frozen=True)
class Rosenbrock:
    """The Rosenbrock function whose minimum is x = [1, ..., 1]"""

    shift: np.ndarray | None = None
    affine: np.ndarray | None = None

    def __post_init__(self) -> None:
        _validate_shift_affine(self.shift, self.affine)


@dataclasses.dataclass(frozen=True)
class Schwefel:
    """The Schwefel function whose minimum is x = [420.9687, ..., 420.9687]"""

    shift: np.ndarray | None = None
    affine: np.ndarray | None = None

    def __post_init__(self) -> None:
        _validate_shift_affine(self.shift, self.affine)


@dataclasses.dataclass(frozen=True)
class Sphere:
    """The sphere function whose minimum is x = [0, ..., 0]"""

    shift: np.ndarray | None = None
    affine: np.ndarray | None = None

    def __post_init__(self) -> None:
        _validate_shift_affine(self.shift, self.affine)


@dataclasses.dataclass(frozen=True)
class Ellipsoid:
    """The Ellipsoid function whose minimum is x = [0, ..., 0]"""

    shift: np.ndarray | None = None
    affine: np.ndarray | None = None

    def __post_init__(self) -> None:
        _validate_shift_affine(self.shift, self.affine)


@dataclasses.dataclass(frozen=True)
class Zakharov:
    """The Zakharov function whose minimum is x = [0, ..., 0]"""

    shift: np.ndarray | None = None
    affine: np.ndarray | None = None

    def __post_init__(self) -> None:
        _validate_shift_affine(self.shift, self.affine)


@dataclasses.dataclass(frozen=True)
class Levy:
    """The Levy function whose minimum is x = [1, ..., 1]"""

    shift: np.ndarray | None = None
    affine: np.ndarray | None = None

    def __post_init__(self) -> None:
        _validate_shift_affine(self.shift, self.affine)


def ackley_func(a: float, b: float, c: float, x: etl.Tensor) -> etl.Tensor:
    return (
        -a * enp.exp(-b * enp.sqrt(enp.mean(x**2, axis=1)))
        - enp.exp(enp.mean(enp.cos(c * x), axis=1))
        + a
        + math.e
    )


def griewank_func(x: etl.Tensor) -> etl.Tensor:
    return (
        1 / 4000 * enp.sum(x**2, axis=1)
        - enp.prod(
            enp.cos(x / enp.sqrt(enp.arange(1, x.shape[1] + 1, dtype=x.dtype))),
            axis=1,
        )
        + 1
    )


def rastrigin_func(x: etl.Tensor) -> etl.Tensor:
    return 10 * x.shape[1] + enp.sum(x**2 - 10 * enp.cos(2 * math.pi * x), axis=1)


def rosenbrock_func(x: etl.Tensor) -> etl.Tensor:
    dim = x.shape[1]
    # torch: x[:, 1:] and x[:, :-1] — expressed with gather (etl's IR slice op
    # cannot express a full-axis slice over the symbolic batch dim).
    x_right = etl.gather(x, enp.arange(1, dim, dtype=np.int32), axis=1)
    x_left = etl.gather(x, enp.arange(0, dim - 1, dtype=np.int32), axis=1)
    return enp.sum(100 * (x_right - x_left**2) ** 2 + (x_left - 1) ** 2, axis=1)


def schwefel_func(x: etl.Tensor) -> etl.Tensor:
    return 418.9828872724338 * x.shape[1] - enp.sum(
        x * enp.sin(enp.sqrt(enp.abs(x))), axis=1
    )


def sphere_func(x: etl.Tensor) -> etl.Tensor:
    return enp.sum(x**2, axis=1)


def ellipsoid_func(x: etl.Tensor) -> etl.Tensor:
    return enp.sum(enp.arange(1, x.shape[1] + 1, dtype=x.dtype) * x**2, axis=1)


def zakharov_func(x: etl.Tensor) -> etl.Tensor:
    d = x.shape[-1]
    i = enp.arange(1, d + 1, dtype=x.dtype)
    sum1 = enp.sum(x**2, axis=-1)
    sum2 = enp.sum(0.5 * i * x, axis=-1)
    return sum1 + sum2**2 + sum2**4


def levy_func(x: etl.Tensor) -> etl.Tensor:
    dim = x.shape[1]
    w = 1 + (x - 1) / 4
    # torch: w[:, 0], w[:, :-1], w[:, -1] — expressed with gather (see rosenbrock_func).
    w_first = etl.gather(w, 0, axis=1)
    w_head = etl.gather(w, enp.arange(0, dim - 1, dtype=np.int32), axis=1)
    w_last = etl.gather(w, dim - 1, axis=1)
    term1 = enp.sin(math.pi * w_first) ** 2
    term2 = enp.sum(
        (w_head - 1) ** 2 * (1 + 10 * enp.sin(math.pi * w_head + 1) ** 2), axis=1
    )
    term3 = (w_last - 1) ** 2 * (1 + enp.sin(2 * math.pi * w_last) ** 2)
    return term1 + term2 + term3


def _shift_affine(
    config: ShiftAffineNumericalProblem, pop: etl.Tensor
) -> etl.Tensor:
    """Apply the config's shift and affine transform to the input points first."""
    dim = pop.shape[1]
    if config.shift is not None:
        shift = etl.constant(etl.tensor(np.asarray(config.shift, dtype=np.float32)))
        pop = pop + shift
    else:
        pop = pop + etl.constant(etl.tensor(np.zeros(dim, dtype=np.float32)))
    if config.affine is not None:
        affine = etl.constant(
            etl.tensor(np.asarray(config.affine, dtype=np.float32))
        )
        pop = etl.matmul(pop, affine)
    return pop


def _true_evaluate(config: object, pop: etl.Tensor) -> etl.Tensor:
    """Dispatch to the true function for the config's type (static at trace time)."""
    if isinstance(config, Ackley):
        return ackley_func(config.a, config.b, config.c, pop)
    if isinstance(config, Griewank):
        return griewank_func(pop)
    if isinstance(config, Rastrigin):
        return rastrigin_func(pop)
    if isinstance(config, Rosenbrock):
        return rosenbrock_func(pop)
    if isinstance(config, Schwefel):
        return schwefel_func(pop)
    if isinstance(config, Sphere):
        return sphere_func(pop)
    if isinstance(config, Ellipsoid):
        return ellipsoid_func(pop)
    if isinstance(config, Zakharov):
        return zakharov_func(pop)
    if isinstance(config, Levy):
        return levy_func(pop)
    raise TypeError(f"unknown numerical problem config: {type(config).__name__}")


def evaluate(
    config: ShiftAffineNumericalProblem,
    problem_state: ProblemState,
    pop: etl.Tensor,
) -> tuple[etl.Tensor, ProblemState]:
    """Evaluate the population: shift/affine the input first, then the true function.

    :param pop: The population of points to evaluate, shape ``(n, dim)`` float32.
    :return: ``(fitness, problem_state)`` with fitness of shape ``(n,)`` float32.
    """
    pop = _shift_affine(config, pop)
    fitness = _true_evaluate(config, pop)
    return fitness, problem_state
