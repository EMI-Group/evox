"""Functional port of the torch evox Opposition-based DE (ODE) algorithm.

Port source: `src/evox/algorithms/so/de_variants/ode.py` (READ-ONLY reference,
math 1:1 per phase). Torch's `ODE.step` runs TWO evaluations: the DE trial
vectors and then, after the first selection, the opposition population
`lb + ub - pop`. The fused `step` owns the whole generation and may call the
opaque `evaluate` closure several times, so both evaluations now fit in ONE
etl step — no phase state machine, generation count matches torch exactly.
`init_step` encodes torch's `init_step` (evaluate the initial population).
Other deviations as in `de.py`: key-based RNG (the DE-mutation draws happen
once per step, in torch's draw order) and array-like config fields normalized
to flat float tuples by the `make_ode` constructor.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, Union

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

__all__ = ["ODE", "ODEState", "init", "init_step", "step", "make_ode"]


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
    """ODE state: population, fitness, the trial vectors of the last step, key.

    (There is no `opposition`/`phase` state: the fused ``step`` evaluates the
    opposition population within the same generation.)
    """

    pop: Tensor
    fit: Tensor
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
    return ODEState(pop=pop, fit=fit, trial_vectors=pop, key=key)


def init_step(config: ODE, state: ODEState, evaluate: Callable[[Tensor], Tensor]) -> ODEState:
    """Initial-generation step (torch ``init_step``): evaluate the initial
    population and record its fitness; no generation is performed yet."""
    fitness = evaluate(state.pop)
    return ODEState(pop=state.pop, fit=fitness, trial_vectors=state.trial_vectors, key=state.key)


def step(config: ODE, state: ODEState, evaluate: Callable[[Tensor], Tensor]) -> ODEState:
    """Run ONE full ODE generation (torch ``step``), i.e. a DE generation
    followed by the opposition-based jump:

    1. Mutation + crossover + clamping (``_de_trial``) → trial vectors.
    2. DE selection: evaluate the trials, keep the better of each individual
       vs its trial.
    3. Opposition-Based Mechanism: build ``opposition = lb + ub - pop`` from
       the POST-selection population (torch order).
    4. Opposition-Based Selection: evaluate the opposites and replace each
       individual whose opposite is better.
    """
    key, subkey = random.split(state.key)
    trial, _ = _de_trial(config, state, subkey)

    # DE selection on the trial vectors.
    fitness = evaluate(trial)
    compare = fitness < state.fit
    pop = etl.select(enp.expand_dims(compare, 1), trial, state.pop)
    fit = etl.select(compare, fitness, state.fit)

    # Opposition-Based Population from the post-selection pop, then selection.
    lb, ub = bake_bounds(config.lb, config.ub, as_row=True)
    opposition = lb + ub - pop
    opposition_fitness = evaluate(opposition)
    compare_opposition = opposition_fitness < fit
    pop = etl.select(enp.expand_dims(compare_opposition, 1), opposition, pop)
    fit = etl.select(compare_opposition, opposition_fitness, fit)

    return ODEState(pop=pop, fit=fit, trial_vectors=trial, key=key)
