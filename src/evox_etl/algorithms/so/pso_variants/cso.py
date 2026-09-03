"""Functional ETL port of the torch evox CSO algorithm.

Plain-function port of ``src/evox/algorithms/so/pso_variants/cso.py``
(read-only torch reference; the ask/tell split follows the old JAX evox
decomposition at tag v0.9.0, with torch semantics winning).  ETL has no eager
mode, so these functions only run inside an active trace via
``etl.build``/``etl.run``.

Config note: ``CSO`` holds numpy boundary arrays, which etl rejects as trace
inputs (numpy arrays are neither TensorSpecs nor static values).  The class is
therefore registered as a zero-child pytree node — the whole config lives in
the tree context and is baked into the graph as constants at compile time
(``etl.run`` validates only its type, not its value, so callers re-passing the
config object is harmless).
"""

from dataclasses import dataclass, replace
from typing import Tuple

import numpy as np

import etl
import etl.numpy as enp
import etl.random as random

from evox_etl.algorithms._operator_shims import clamp

Tensor = etl.SymbolicTensor


@dataclass(frozen=True)
class CSO:
    """CSO hyperparameters (torch ``CSO.__init__`` minus ``device``)."""

    pop_size: int
    lb: np.ndarray
    ub: np.ndarray
    phi: float = 0.0
    mean: np.ndarray | None = None
    stdev: np.ndarray | None = None


# Register the numpy-holding config as an opaque pytree leaf (see module
# docstring): flatten -> no children, unflatten -> the object itself.
etl.register_pytree_node(CSO, lambda c: ((), c), lambda c, _children: c)


@dataclass(frozen=True)
class CSOState:
    """CSO mutable state; ``students`` records the rows updated by the last ask."""

    pop: Tensor
    velocity: Tensor
    fit: Tensor
    students: Tensor
    key: Tensor


def _bake(arr: np.ndarray) -> Tensor:
    """Bake a config numpy array into a (1, dim) float32 graph constant."""
    return etl.ops.constant(etl.core.tensor(np.asarray(arr, dtype=np.float32)[None, :]))


def init(config: CSO, key: Tensor) -> CSOState:
    """Draw the initial population, velocity and placeholder leaves."""
    pop_size = config.pop_size
    dim = config.lb.shape[0]
    lb = _bake(config.lb)
    ub = _bake(config.ub)
    length = ub - lb

    state_key, init_key = random.split(key)
    key_pop, key_vel = random.split(init_key)

    if config.mean is not None and config.stdev is not None:
        mean_c = _bake(config.mean)
        stdev_c = _bake(config.stdev)
        pop = clamp(
            mean_c
            + stdev_c
            * random.normal(key_pop, (pop_size, dim), 0.0, 1.0, etl.float32),
            lb,
            ub,
        )
    else:
        pop = (
            length
            * random.uniform(key_pop, (pop_size, dim), 0.0, 1.0, etl.float32)
            + lb
        )
    velocity = (
        2 * length * random.uniform(key_vel, (pop_size, dim), 0.0, 1.0, etl.float32)
        - length
    )
    return CSOState(
        pop=pop,
        velocity=velocity,
        fit=enp.zeros((pop_size,), dtype=etl.float32),
        students=enp.zeros((pop_size // 2,), dtype=etl.int64),
        key=state_key,
    )


def init_ask(config: CSO, state: CSOState) -> Tuple[Tensor, CSOState]:
    """First-generation candidates: the whole population."""
    return state.pop, state


def init_tell(config: CSO, state: CSOState, fitness: Tensor) -> CSOState:
    """Store the first-generation fitness."""
    return replace(state, fit=fitness)


def ask(config: CSO, state: CSOState) -> Tuple[Tensor, CSOState]:
    """Pair particles at random and update the losers (students) CSO-style."""
    pop_size = config.pop_size
    dim = config.lb.shape[0]
    half = pop_size // 2
    lb = _bake(config.lb)
    ub = _bake(config.ub)
    vel_range = ub - lb

    key, subkey = random.split(state.key)
    perm = random.permutation(subkey, pop_size, dtype=etl.int64)
    left = perm[:half]
    right = perm[half:]
    mask = etl.gather(state.fit, left, axis=0) < etl.gather(state.fit, right, axis=0)
    teachers = etl.select(mask, left, right)
    students = etl.select(mask, right, left)
    center = etl.mean(state.pop, axes=0)

    key, subkey2 = random.split(key)
    lambdas = random.uniform(subkey2, (3, half, dim), 0.0, 1.0, etl.float32)
    lambda1, lambda2, lambda3 = lambdas[0], lambdas[1], lambdas[2]

    pop_students = etl.gather(state.pop, students, axis=0)
    pop_teachers = etl.gather(state.pop, teachers, axis=0)
    student_velocity = (
        lambda1 * etl.gather(state.velocity, students, axis=0)  # inertia
        + lambda2 * (pop_teachers - pop_students)  # learn from teachers
        + config.phi * lambda3 * (center - pop_students)  # converge to the center
    )
    student_velocity = clamp(student_velocity, -vel_range, vel_range)
    candidates = clamp(pop_students + student_velocity, lb, ub)
    new_pop = etl.scatter(state.pop, students, candidates, axis=0)

    return candidates, replace(state, pop=new_pop, students=students, key=key)


def tell(config: CSO, state: CSOState, fitness: Tensor) -> CSOState:
    """Scatter the evaluated candidates' fitness onto the students' rows."""
    return replace(state, fit=etl.scatter(state.fit, state.students, fitness, axis=0))
