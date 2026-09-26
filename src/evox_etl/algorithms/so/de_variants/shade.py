"""Functional ETL port of the torch evox SHADE algorithm.

Port source: ``src/evox/algorithms/so/de_variants/shade.py`` (1:1 semantics).
``step`` owns ONE full generation: it draws F/CR from the success-history
memory, builds the trial vectors, evaluates them through the opaque
``evaluate`` closure, then performs the selection and the memory update from
the fitnesses. RNG is key-based (``etl.random``): the key is advanced in the
state, one ``random.split`` per random op in torch's exact draw order; the
selection/memory half draws no randomness. Bounds are baked once per function
as (1, dim) constants; ``lb``/``ub`` config fields are normalized to flat
tuples of plain Python floats by the ``make_shade`` constructor.
"""

from dataclasses import dataclass
from typing import Any, Callable

import etl
import etl.numpy as enp
import etl.random as random

from evox_etl.algorithms._config_utils import (
    ArrayLike,
    bake_float32_constant,
    normalize_bounds,
    require_ge,
)
from evox_etl.operators.jit_fix_operator import clamp
from evox_etl.operators.crossover import DE_binary_crossover, DE_differential_sum
from evox_etl.operators.selection import select_rand_pbest

Tensor = etl.SymbolicTensor


@dataclass(frozen=True)
class SHADE:
    """SHADE hyperparameters (mirrors torch ``SHADE.__init__``, device dropped).

    Dumb frozen dataclass — construct via ``make_shade``, which validates and
    normalizes the bounds to flat float tuples before construction.

    :param pop_size: Population size (>= 9).
    :param lb: Lower bounds of the search space (1-D array/tuple).
    :param ub: Upper bounds of the search space (same shape as ``lb``).
    :param diff_padding_num: Number of differential padding vectors (default 9).
    """

    pop_size: int
    lb: Any
    ub: Any
    diff_padding_num: int = 9


def make_shade(pop_size: int, lb: ArrayLike, ub: ArrayLike, diff_padding_num: int = 9) -> SHADE:
    """Build a :class:`SHADE` config, validating hyperparameters and normalizing bounds.

    Same validation semantics as torch ``SHADE.__init__``'s asserts, raised as ValueError.
    """
    require_ge("pop_size", pop_size, 9)
    lb, ub = normalize_bounds(lb, ub)
    return SHADE(pop_size=pop_size, lb=lb, ub=ub, diff_padding_num=diff_padding_num)


@dataclass(frozen=True)
class SHADEState:
    """SHADE algorithm state; every leaf is an etl tensor.

    ``trial_vectors``/``F_vect``/``CR_vect`` record the last generation's
    candidates and their per-individual parameters (observable state); the
    memory update consumes them within the same ``step`` call.
    """

    best_index: Tensor
    Memory_FCR: Tensor
    pop: Tensor
    fit: Tensor
    trial_vectors: Tensor
    F_vect: Tensor
    CR_vect: Tensor
    key: Tensor


def init(config: SHADE, key: Tensor) -> SHADEState:
    """Initialize the population (torch uses randn scaled by bounds — no clamp)."""
    pop_size, dim = config.pop_size, len(config.lb)
    lb = bake_float32_constant(config.lb, shape=(1, -1))
    ub = bake_float32_constant(config.ub, shape=(1, -1))
    key, subkey = random.split(key)
    pop = random.normal(subkey, (pop_size, dim), 0.0, 1.0, "float32") * (ub - lb) + lb
    return SHADEState(
        best_index=enp.full((), 0, dtype="int64"),
        Memory_FCR=enp.full((2, pop_size), 0.5, dtype="float32"),
        pop=pop,
        fit=enp.full((pop_size,), float("inf"), dtype="float32"),
        trial_vectors=pop,
        F_vect=enp.zeros((pop_size,), dtype="float32"),
        CR_vect=enp.zeros((pop_size,), dtype="float32"),
        key=key,
    )


def step(config: SHADE, state: SHADEState, evaluate: Callable[[Tensor], Tensor]) -> SHADEState:
    """Run ONE full SHADE generation (torch ``step``): trial-vector generation,
    evaluation, selection, then success-history memory update. No randomness
    is drawn after the candidate generation."""
    pop_size = config.pop_size
    lb = bake_float32_constant(config.lb, shape=(1, -1))
    ub = bake_float32_constant(config.ub, shape=(1, -1))

    # (1) Random permutation of memory indices (argsort-of-rand, torch-exact).
    key, k_fcr = random.split(state.key)
    FCR_ids = etl.argsort(
        random.uniform(k_fcr, (pop_size,), 0.0, 1.0, "float32"), axis=0, stable=True
    )
    M_F_vect = etl.gather(state.Memory_FCR[0], FCR_ids, axis=0)
    M_CR_vect = etl.gather(state.Memory_FCR[1], FCR_ids, axis=0)

    # (2) F from the success-history memory, clamped to [0, 1].
    key, k_f = random.split(key)
    F_vect = random.normal(k_f, (pop_size,), 0.0, 1.0, "float32") * 0.1 + M_F_vect
    F_vect = clamp(
        F_vect, enp.zeros((pop_size,), dtype="float32"), enp.ones((pop_size,), dtype="float32")
    )

    # (3) CR likewise.
    key, k_cr = random.split(key)
    CR_vect = random.normal(k_cr, (pop_size,), 0.0, 1.0, "float32") * 0.1 + M_CR_vect
    CR_vect = clamp(
        CR_vect, enp.zeros((pop_size,), dtype="float32"), enp.ones((pop_size,), dtype="float32")
    )

    # (4) Difference vectors (num_diff_vectors is the constant 1 in torch).
    key, k_ds = random.split(key)
    difference_sum, _ = DE_differential_sum(
        k_ds,
        config.diff_padding_num,
        enp.ones((pop_size,), dtype="int32"),
        enp.arange(pop_size, dtype="int32"),
        state.pop,
    )

    # (5)-(6) current-to-pbest mutation.
    key, k_pbest = random.split(key)
    pbest_vect = select_rand_pbest(k_pbest, 0.05, state.pop, state.fit)
    current_vect = state.pop
    base_vector = current_vect + enp.expand_dims(F_vect, 1) * (pbest_vect - current_vect)
    mutation_vector = base_vector + difference_sum * enp.expand_dims(F_vect, 1)

    # (7)-(8) Binary crossover and clamp into bounds.
    key, k_bin = random.split(key)
    trial_vectors = DE_binary_crossover(k_bin, mutation_vector, current_vect, CR_vect)
    trial_vectors = clamp(trial_vectors, lb, ub)

    # (9) Selection over the trial vectors.
    fitness = evaluate(trial_vectors)
    compare = fitness < state.fit
    pop = etl.select(enp.expand_dims(compare, 1), trial_vectors, state.pop)
    fit = etl.select(compare, fitness, state.fit)
    best_index = etl.argmin(fit)

    # torch computes deltas AFTER updating self.fit — use the updated fit.
    deltas = fit - fitness

    S_F = enp.full((pop_size,), float("nan"), dtype="float32")
    S_CR = enp.full((pop_size,), float("nan"), dtype="float32")
    S_delta = enp.full((pop_size,), float("nan"), dtype="float32")

    for i in range(pop_size):  # get_success_delta
        is_success = etl.cast(compare[i], "float32")
        F = F_vect[i]
        CR = CR_vect[i]
        delta = deltas[i]

        S_F_update_temp = etl.roll(S_F, 1, axis=0)
        S_F_update = etl.concatenate([enp.expand_dims(F, 0), S_F_update_temp[1:]], axis=0)

        S_CR_update_temp = etl.roll(S_CR, 1, axis=0)
        S_CR_update = etl.concatenate([enp.expand_dims(CR, 0), S_CR_update_temp[1:]], axis=0)

        S_delta_update_temp = etl.roll(S_delta, 1, axis=0)
        S_delta_update = etl.concatenate(
            [enp.expand_dims(delta, 0), S_delta_update_temp[1:]], axis=0
        )

        S_F = is_success * S_F_update + (1.0 - is_success) * S_F_update_temp
        S_CR = is_success * S_CR_update + (1.0 - is_success) * S_CR_update_temp
        S_delta = is_success * S_delta_update + (1.0 - is_success) * S_delta_update_temp

    norm_delta = S_delta / etl.nansum(S_delta, axes=0)
    M_CR = etl.nansum(norm_delta * S_CR, axes=0)
    M_F = etl.nansum(norm_delta * (S_F**2), axes=0) / etl.nansum(norm_delta * S_F, axes=0)

    Memory_FCR_update = etl.roll(state.Memory_FCR, 1, axis=1)
    # Memory_FCR_update[0, 0] = M_F via a boolean mask (no index assignment).
    mask00 = enp.logical_and(
        enp.expand_dims(enp.arange(2, dtype="int32") == 0, 1),
        enp.expand_dims(enp.arange(pop_size, dtype="int32") == 0, 0),
    )
    Memory_FCR_update = etl.select(mask00, M_F, Memory_FCR_update)
    # Memory_FCR_update[1, 0] = M_CR.
    mask10 = enp.logical_and(
        enp.expand_dims(enp.arange(2, dtype="int32") == 1, 1),
        enp.expand_dims(enp.arange(pop_size, dtype="int32") == 0, 0),
    )
    Memory_FCR_update = etl.select(mask10, M_CR, Memory_FCR_update)

    is_F_nan = etl.isnan(M_F)
    Memory_FCR_update = etl.select(is_F_nan, state.Memory_FCR, Memory_FCR_update)

    # torch's is_S_nan = torch.all(torch.isnan(compare)) is statically False
    # (isnan of a bool tensor), so the final Memory_FCR is Memory_FCR_update.
    Memory_FCR = Memory_FCR_update

    return SHADEState(
        best_index=best_index,
        Memory_FCR=Memory_FCR,
        pop=pop,
        fit=fit,
        trial_vectors=trial_vectors,
        F_vect=F_vect,
        CR_vect=CR_vect,
        key=key,
    )
