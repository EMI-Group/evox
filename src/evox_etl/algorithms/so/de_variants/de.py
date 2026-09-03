"""Functional port of the torch evox Differential Evolution (DE) algorithm.

Port source: `src/evox/algorithms/so/de_variants/de.py` (READ-ONLY reference,
semantics 1:1). Torch's `step` calls `self.evaluate` in the middle of the
method, so the functional split point is there: `ask` runs the mutation +
crossover half (via `_de_trial`) and `tell` runs the selection half.
`init_ask`/`init_tell` encode torch's `init_step` (initial-population
evaluation before the first generation).

Deviations from torch:
- RNG: torch's global `torch.rand`/`torch.randint` draws are key-based
  (`etl.random`), one split per draw in torch's draw order; keys live in the
  state and advance there. `tell` never draws randomness.
- Array-like config fields (`lb`/`ub`/`mean`/`stdev` and a tuple
  `differential_weight`) are normalized to tuples of plain Python floats in
  `__post_init__`, because `etl.build` rejects numpy arrays as static args.
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np

import etl
import etl.numpy as enp
import etl.random as random

from evox_etl.algorithms._operator_shims import clamp

Tensor = etl.SymbolicTensor

__all__ = ["DE", "DEState", "init", "init_ask", "init_tell", "ask", "tell"]


def _to_float_tuple(value) -> Tuple[float, ...]:
    """Normalize an array-like value to a tuple of plain Python floats."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    return tuple(float(v) for v in value)


def _constant_1d(values) -> Tensor:
    """Bake a 1-D float32 numpy array as a graph constant."""
    return etl.ops.constant(etl.core.tensor(np.asarray(values, dtype=np.float32)))


@dataclass(frozen=True)
class DE:
    """Differential Evolution (DE) config, mirroring torch `DE.__init__` (device dropped)."""

    pop_size: int
    lb: Union[Tuple[float, ...], Any]
    ub: Union[Tuple[float, ...], Any]
    base_vector: str = "rand"
    num_difference_vectors: int = 1
    differential_weight: Union[float, tuple] = 0.5
    cross_probability: float = 0.9
    mean: Optional[Any] = None
    stdev: Optional[Any] = None

    def __post_init__(self) -> None:
        assert self.pop_size >= 4
        assert 0 < self.cross_probability <= 1
        assert 1 <= self.num_difference_vectors < self.pop_size // 2
        assert self.base_vector in ["rand", "best"]
        assert len(self.lb) == len(self.ub)
        if self.num_difference_vectors == 1:
            assert isinstance(self.differential_weight, float)
        else:
            assert not isinstance(self.differential_weight, float)
            assert len(self.differential_weight) == self.num_difference_vectors
        object.__setattr__(self, "lb", _to_float_tuple(self.lb))
        object.__setattr__(self, "ub", _to_float_tuple(self.ub))
        if self.mean is not None:
            object.__setattr__(self, "mean", _to_float_tuple(self.mean))
        if self.stdev is not None:
            object.__setattr__(self, "stdev", _to_float_tuple(self.stdev))
        if not isinstance(self.differential_weight, float):
            object.__setattr__(self, "differential_weight", _to_float_tuple(self.differential_weight))


@dataclass(frozen=True)
class DEState:
    """DE state: population, fitness, pending trial vectors and the RNG key."""

    pop: Tensor
    fit: Tensor
    trial_vectors: Tensor
    key: Tensor


def _bounds(config: DE) -> Tuple[Tensor, Tensor]:
    """Bake the (1, dim) lower/upper bound constants (torch's `lb[None, :]`)."""
    lb = enp.reshape(_constant_1d(config.lb), (1, -1))
    ub = enp.reshape(_constant_1d(config.ub), (1, -1))
    return lb, ub


def _de_trial(config: DE, state: DEState, subkey: Tensor) -> Tuple[Tensor, Tensor]:
    """Port of torch `DE.step`'s mutation + crossover half (up to `evaluate`).

    Draws linearly from `subkey` in torch's order: `num_vec` randint index
    draws (torch draws all `num_vec` even in best mode), then the crossover
    probability draw and the forced-crossover-dimension draw. Returns the
    clamped trial vectors and the last advanced subkey.
    """
    pop, fit = state.pop, state.fit
    pop_size, dim = config.pop_size, len(config.lb)
    num_vec = config.num_difference_vectors * 2 + (0 if config.base_vector == "best" else 1)

    choices = []
    for _ in range(num_vec):
        subkey, k = random.split(subkey)
        choices.append(random.randint(k, (pop_size,), 0, pop_size, "int32"))

    # Determine the base vector.
    if config.base_vector == "best":
        best_index = etl.argmin(fit)
        base = enp.expand_dims(etl.gather(pop, best_index, axis=0), 0)
        start_index = 0
    else:
        base = etl.gather(pop, choices[0], axis=0)
        start_index = 1

    # Sum of difference vectors from pairs of randomly chosen individuals.
    difference_vector = etl.sum(
        etl.stack(
            [
                etl.gather(pop, choices[i], axis=0) - etl.gather(pop, choices[i + 1], axis=0)
                for i in range(start_index, num_vec - 1, 2)
            ],
            axis=0,
        ),
        axes=0,
    )

    # Mutant vector: base + F * difference (torch applies F after the sum).
    if isinstance(config.differential_weight, float):
        new_pop = base + config.differential_weight * difference_vector
    else:
        f_const = _constant_1d(config.differential_weight)
        new_pop = base + f_const * difference_vector

    # Crossover: take a dim from the mutant with prob CR, plus one forced dim.
    subkey, kc = random.split(subkey)
    cross_prob = random.uniform(kc, (pop_size, dim), 0.0, 1.0, "float32") < config.cross_probability
    subkey, kd = random.split(subkey)
    random_dim = random.randint(kd, (pop_size, 1), 0, dim, "int32")
    mask = etl.logical_or(cross_prob, enp.expand_dims(enp.arange(dim, dtype="int32"), 0) == random_dim)
    new_pop = etl.select(mask, new_pop, pop)

    # Ensure the trial population is within bounds.
    lb, ub = _bounds(config)
    new_pop = clamp(new_pop, lb, ub)
    return new_pop, subkey


def init(config: DE, key: Tensor) -> DEState:
    """Draw the initial population (normal around `mean`/`stdev`, else uniform in bounds)."""
    key, subkey = random.split(key)
    pop_size, dim = config.pop_size, len(config.lb)
    lb, ub = _bounds(config)
    if config.mean is not None and config.stdev is not None:
        mean_c = _constant_1d(config.mean)
        stdev_c = _constant_1d(config.stdev)
        pop = mean_c + stdev_c * random.normal(subkey, (pop_size, dim), 0.0, 1.0, "float32")
        pop = clamp(pop, lb, ub)
    else:
        pop = random.uniform(subkey, (pop_size, dim), 0.0, 1.0, "float32") * (ub - lb) + lb
    fit = enp.full((pop_size,), float("inf"), dtype="float32")
    return DEState(pop=pop, fit=fit, trial_vectors=pop, key=key)


def init_ask(config: DE, state: DEState) -> Tuple[Tensor, DEState]:
    """Return the initial population for the workflow's first evaluation."""
    return state.pop, state


def init_tell(config: DE, state: DEState, fitness: Tensor) -> DEState:
    """Record the fitness of the initial population (torch init_step)."""
    return DEState(pop=state.pop, fit=fitness, trial_vectors=state.trial_vectors, key=state.key)


def ask(config: DE, state: DEState) -> Tuple[Tensor, DEState]:
    """Produce DE trial vectors (mutation + crossover) for evaluation."""
    key, subkey = random.split(state.key)
    trial, _ = _de_trial(config, state, subkey)
    return trial, DEState(pop=state.pop, fit=state.fit, trial_vectors=trial, key=key)


def tell(config: DE, state: DEState, fitness: Tensor) -> DEState:
    """Keep the better of each individual vs its trial vector (torch's selection half)."""
    compare = fitness < state.fit
    pop = etl.select(enp.expand_dims(compare, 1), state.trial_vectors, state.pop)
    fit = etl.select(compare, fitness, state.fit)
    return DEState(pop=pop, fit=fit, trial_vectors=state.trial_vectors, key=state.key)
