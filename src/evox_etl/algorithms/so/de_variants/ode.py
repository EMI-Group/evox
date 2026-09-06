"""Functional port of the torch evox Opposition-based DE (ODE) algorithm.

Port source: `src/evox/algorithms/so/de_variants/ode.py` (READ-ONLY reference,
math 1:1 per phase). Torch's `ODE.step` runs TWO evaluations: the DE trial
vectors and then, after the first selection, the opposition population
`lb + ub - pop`. The functional one-evaluate-per-generation contract
(`ask` -> evaluate -> `tell`) cannot fit two evaluates into one step, so one
torch ODE step becomes TWO etl generations via a phase state machine:
- phase 0: `ask` returns the DE trial vectors; `tell` performs the DE
  selection and computes the opposition `lb + ub - pop` (post-selection `pop`,
  like torch), then switches to phase 1.
- phase 1: `ask` returns the pending opposition; `tell` performs the
  opposition selection and switches back to phase 0.
The generation count therefore doubles relative to torch. `init_ask`/
`init_tell` encode torch's `init_step` (initial-population evaluation before
the first generation). Other deviations as
in `de.py`: key-based RNG (ask advances the key even in phase 1, since both
branches execute) and array-like config fields normalized to flat float tuples
by the `make_ode` constructor.
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import etl
import etl.numpy as enp
import etl.random as random

from evox_etl.algorithms._config_utils import (
    ArrayLike,
    bake_bounds,
    bake_float32_constant,
    normalize_bounds,
    require_between,
    require_choice,
    require_ge,
    to_float_tuple,
)
from evox_etl.algorithms.so.de_variants.de import _de_trial
from evox_etl.operators.jit_fix_operator import clamp

Tensor = etl.SymbolicTensor

__all__ = ["ODE", "ODEState", "init", "init_ask", "init_tell", "ask", "tell", "make_ode"]


@dataclass(frozen=True)
class ODE:
    """Opposition-based DE (ODE) config, mirroring torch `ODE.__init__` (device dropped).

    Dumb frozen dataclass — construct via `make_ode`, which validates and
    normalizes array-like fields to flat float tuples before construction.
    """

    pop_size: int
    lb: Union[Tuple[float, ...], Any]
    ub: Union[Tuple[float, ...], Any]
    base_vector: str = "rand"
    num_difference_vectors: int = 1
    differential_weight: Union[float, tuple] = 0.5
    cross_probability: float = 0.9
    mean: Optional[Any] = None
    stdev: Optional[Any] = None


def make_ode(
    pop_size: int,
    lb: ArrayLike,
    ub: ArrayLike,
    base_vector: str = "rand",
    num_difference_vectors: int = 1,
    differential_weight: float | tuple = 0.5,
    cross_probability: float = 0.9,
    mean: ArrayLike | None = None,
    stdev: ArrayLike | None = None,
) -> ODE:
    """Build an :class:`ODE` config, validating hyperparameters and normalizing array-like fields.

    Same validation semantics as torch ``ODE.__init__``'s asserts, raised as ValueError.
    """
    require_ge("pop_size", pop_size, 4)
    require_between("cross_probability", cross_probability, 0, 1, low_inclusive=False)
    require_between(
        "num_difference_vectors", num_difference_vectors, 1, pop_size // 2, high_inclusive=False
    )
    require_choice("base_vector", base_vector, ["rand", "best"])
    lb, ub = normalize_bounds(lb, ub)
    if num_difference_vectors == 1:
        # np.float64 subclasses float, so require the exact Python type: numpy
        # scalars are not valid graph operands and must be rejected.
        if type(differential_weight) is not float:
            raise ValueError(
                "differential_weight must be a float when num_difference_vectors == 1, "
                f"got {differential_weight!r}"
            )
    else:
        if type(differential_weight) is float:
            raise ValueError(
                "differential_weight must be a sequence (not a float) when "
                f"num_difference_vectors > 1, got {differential_weight!r}"
            )
        differential_weight = to_float_tuple(differential_weight)
        if len(differential_weight) != num_difference_vectors:
            raise ValueError(
                "differential_weight must have length num_difference_vectors "
                f"({num_difference_vectors}), got length {len(differential_weight)}"
            )
    return ODE(
        pop_size=pop_size,
        lb=lb,
        ub=ub,
        base_vector=base_vector,
        num_difference_vectors=num_difference_vectors,
        differential_weight=differential_weight,
        cross_probability=cross_probability,
        mean=to_float_tuple(mean) if mean is not None else None,
        stdev=to_float_tuple(stdev) if stdev is not None else None,
    )


@dataclass(frozen=True)
class ODEState:
    """ODE state: population, fitness, pending opposition, phase, trial vectors, key."""

    pop: Tensor
    fit: Tensor
    opposition: Tensor
    phase: Tensor
    trial_vectors: Tensor
    key: Tensor


def init(config: ODE, key: Tensor) -> ODEState:
    """Draw the initial population (normal around `mean`/`stdev`, else uniform in bounds)."""
    key, subkey = random.split(key)
    pop_size, dim = config.pop_size, len(config.lb)
    lb, ub = bake_bounds(config.lb, config.ub, as_row=True)
    if config.mean is not None and config.stdev is not None:
        mean_c = bake_float32_constant(config.mean)
        stdev_c = bake_float32_constant(config.stdev)
        pop = mean_c + stdev_c * random.normal(subkey, (pop_size, dim), 0.0, 1.0, "float32")
        pop = clamp(pop, lb, ub)
    else:
        pop = random.uniform(subkey, (pop_size, dim), 0.0, 1.0, "float32") * (ub - lb) + lb
    fit = enp.full((pop_size,), float("inf"), dtype="float32")
    phase = enp.full((), 0, dtype="int32")
    return ODEState(pop=pop, fit=fit, opposition=pop, phase=phase, trial_vectors=pop, key=key)


def init_ask(config: ODE, state: ODEState) -> Tuple[Tensor, ODEState]:
    """Return the initial population for the workflow's first evaluation."""
    return state.pop, state


def init_tell(config: ODE, state: ODEState, fitness: Tensor) -> ODEState:
    """Record the fitness of the initial population (torch init_step)."""
    return ODEState(
        pop=state.pop,
        fit=fitness,
        opposition=state.opposition,
        phase=state.phase,
        trial_vectors=state.trial_vectors,
        key=state.key,
    )


def ask(config: ODE, state: ODEState) -> Tuple[Tensor, ODEState]:
    """Phase 0: DE trial vectors; phase 1: the pending opposition population.

    Both branches execute (pure), so the key advances by the trial draws even
    in phase 1; the returned phase is unchanged.
    """
    key, subkey = random.split(state.key)
    trial, _ = _de_trial(config, state, subkey)
    candidates = etl.select(state.phase == 0, trial, state.opposition)
    return candidates, ODEState(
        pop=state.pop,
        fit=state.fit,
        opposition=state.opposition,
        phase=state.phase,
        trial_vectors=trial,
        key=key,
    )


def tell(config: ODE, state: ODEState, fitness: Tensor) -> ODEState:
    """Phase 0: DE selection + compute opposition; phase 1: opposition selection."""
    # Phase 0 branch: DE selection on the trial vectors, then opposition.
    compare_de = fitness < state.fit
    pop0 = etl.select(enp.expand_dims(compare_de, 1), state.trial_vectors, state.pop)
    fit0 = etl.select(compare_de, fitness, state.fit)
    lb, ub = bake_bounds(config.lb, config.ub, as_row=True)
    opposition0 = lb + ub - pop0
    phase0 = enp.full((), 1, dtype="int32")

    # Phase 1 branch: opposition-based selection.
    compare_opp = fitness < state.fit
    pop1 = etl.select(enp.expand_dims(compare_opp, 1), state.opposition, state.pop)
    fit1 = etl.select(compare_opp, fitness, state.fit)
    phase1 = enp.full((), 0, dtype="int32")

    # Pick the branch fields by phase; the key is unchanged (tell draws nothing).
    phase_cond = state.phase == 0
    pop = etl.select(phase_cond, pop0, pop1)
    fit = etl.select(phase_cond, fit0, fit1)
    opposition = etl.select(phase_cond, opposition0, state.opposition)
    phase = etl.select(phase_cond, phase0, phase1)
    return ODEState(
        pop=pop,
        fit=fit,
        opposition=opposition,
        phase=phase,
        trial_vectors=state.trial_vectors,
        key=state.key,
    )
