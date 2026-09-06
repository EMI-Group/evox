"""Functional ETL port of the torch evox CoDE algorithm (composite DE).

Port source: ``src/evox/algorithms/so/de_variants/code.py`` (READ-ONLY torch
reference, ported 1:1). Torch's single ``step`` evaluates the stacked batch of
``3 * pop_size`` trial vectors in one ``self.evaluate`` call; here that splits
at the evaluate boundary: ``ask`` builds the clamped ``(3 * pop_size, dim)``
trial batch and ``tell`` applies the per-individual strategy selection plus
the population update. RNG is key-based (``etl.random``): ``ask`` splits the
state key per random op in torch's draw order and stores the advanced key;
``tell`` never draws randomness. Config ``lb``/``ub``/``param_pool`` are
normalized to tuples of plain Python floats by the ``make_code`` constructor
(``param_pool`` stays a tuple of (F, CR) pairs).
"""

from dataclasses import dataclass

import numpy as np

import etl
import etl.numpy as enp
import etl.random as random
from etl.core import SymbolicTensor

from evox_etl.algorithms._config_utils import ArrayLike, bake_float32_constant, to_float_tuple
from evox_etl.operators.jit_fix_operator import _take_along_axis, clamp
from evox_etl.operators.crossover import (
    DE_arithmetic_recombination,
    DE_binary_crossover,
    DE_differential_sum,
    DE_exponential_crossover,
)
from evox_etl.operators.selection import select_rand_pbest

# Strategy codes (4 bits): [base_vec_prim, base_vec_sec, diff_num, cross_strategy]
# base_vec      : 0="rand", 1="best", 2="pbest", 3="current"
# cross_strategy: 0=bin   , 1=exp   , 2=arith
# (torch's rand_1_bin, rand_2_bin, current2rand_1 — plain Python ints used as
# STATIC values inside traces).
STRATEGIES = ((0, 0, 1, 0), (0, 0, 2, 0), (0, 0, 1, 2))


@dataclass(frozen=True)
class CoDE:
    """CoDE config; mirrors the torch ``CoDE.__init__`` signature (no device).

    Reference: Wang Y, Cai Z, Zhang Q. Differential evolution with composite
    trial vector generation strategies and control parameters. IEEE TEVC, 2011.

    :param pop_size: The size of the population.
    :param lb: The lower bounds of the search space (1-D array-like).
    :param ub: The upper bounds of the search space (1-D array-like).
    :param diff_padding_num: The number of differential padding vectors to use.
    :param param_pool: Control parameter pairs (F, CR) tried per strategy.
    :param replace: Kept for API parity only — torch CoDE never passes it to
        ``DE_differential_sum`` in ``step``, so it is unused here too.

    Dumb frozen dataclass — construct via ``make_code``, which normalizes the
    array-like fields (torch CoDE performs no validation) before construction.
    """

    pop_size: int
    lb: tuple[float, ...]
    ub: tuple[float, ...]
    diff_padding_num: int = 5
    param_pool: tuple[tuple[float, float], ...] = (
        (1.0, 0.1),
        (1.0, 0.9),
        (0.8, 0.2),
    )
    replace: bool = False


def make_code(
    pop_size: int,
    lb: ArrayLike,
    ub: ArrayLike,
    diff_padding_num: int = 5,
    param_pool: ArrayLike = ((1.0, 0.1), (1.0, 0.9), (0.8, 0.2)),
    replace: bool = False,
) -> CoDE:
    """Build a :class:`CoDE` config, normalizing array-like fields to plain float tuples.

    Torch CoDE has no validation — mirrors that; ``param_pool`` is stored as a
    tuple of (F, CR) pairs (its natural (3, 2) shape is kept).
    """
    return CoDE(
        pop_size=pop_size,
        lb=to_float_tuple(lb),
        ub=to_float_tuple(ub),
        diff_padding_num=diff_padding_num,
        param_pool=tuple(to_float_tuple(pair) for pair in param_pool),
        replace=replace,
    )


@dataclass(frozen=True)
class CoDEState:
    """CoDE algorithm state (leaves are etl tensors only).

    ``trial_vectors`` holds the ``(3 * pop_size, dim)`` stacked candidates
    pending evaluation (written by ``ask``, read by ``tell``).
    """

    best_index: SymbolicTensor  # () int64
    pop: SymbolicTensor  # (pop_size, dim) float32
    fit: SymbolicTensor  # (pop_size,) float32
    trial_vectors: SymbolicTensor  # (3 * pop_size, dim) float32
    key: SymbolicTensor  # () int64


def init(config: CoDE, key: SymbolicTensor) -> CoDEState:
    """Create the initial state: torch randn population (no clamp, 1:1 port),
    +inf fitness, best_index 0 and an empty trial-vector batch."""
    pop_size = config.pop_size
    dim = len(config.lb)
    lb = bake_float32_constant(config.lb, shape=(1, -1))
    ub = bake_float32_constant(config.ub, shape=(1, -1))

    key, subkey = random.split(key)
    pop = random.normal(subkey, (pop_size, dim), 0.0, 1.0, "float32") * (ub - lb) + lb
    fit = enp.full((pop_size,), float("inf"), dtype="float32")
    best_index = enp.full((), 0, dtype="int64")
    trial_vectors = enp.zeros((3 * pop_size, dim), dtype="float32")
    return CoDEState(
        best_index=best_index,
        pop=pop,
        fit=fit,
        trial_vectors=trial_vectors,
        key=key,
    )


def ask(config: CoDE, state: CoDEState) -> tuple[SymbolicTensor, CoDEState]:
    """Build the clamped ``(3 * pop_size, dim)`` trial batch (one trial vector
    per strategy per individual) and return it with the advanced state."""
    pop_size = config.pop_size
    dim = len(config.lb)
    lb = bake_float32_constant(config.lb, shape=(1, -1))
    ub = bake_float32_constant(config.ub, shape=(1, -1))
    param_pool_c = bake_float32_constant(config.param_pool)  # (3, 2) float32
    indices = enp.arange(pop_size, dtype="int32")

    key, k_params = random.split(state.key)
    param_ids = random.randint(k_params, (3, pop_size), 0, 3, "int32")
    params = etl.gather(param_pool_c, param_ids, axis=0)  # (3, pop_size, 2)
    differential_weight = params[:, :, 0]  # (3, pop_size)
    cross_probability = params[:, :, 1]  # (3, pop_size)

    trials = []
    for i in range(3):
        num_diff = etl.ops.constant(
            etl.core.tensor(np.full((pop_size,), STRATEGIES[i][2], np.int32))
        )
        key, k_diff = random.split(key)
        difference_sum, rand_vec_idx = DE_differential_sum(
            k_diff, config.diff_padding_num, num_diff, indices, state.pop
        )
        rand_vec = etl.gather(state.pop, rand_vec_idx, axis=0)
        best_vec = etl.tile(
            enp.expand_dims(etl.gather(state.pop, state.best_index, axis=0), 0),
            (pop_size, 1),
        )
        key, k_pbest = random.split(key)
        pbest_vec = select_rand_pbest(k_pbest, 0.05, state.pop, state.fit)
        current_vec = state.pop

        vec_merge = etl.stack([rand_vec, best_vec, pbest_vec, current_vec], axis=0)
        base_vec_prim = vec_merge[STRATEGIES[i][0]]
        base_vec_sec = vec_merge[STRATEGIES[i][1]]
        base_vec = base_vec_prim + enp.expand_dims(differential_weight[i], 1) * (
            base_vec_sec - base_vec_prim
        )
        mutation_vec = base_vec + difference_sum * enp.expand_dims(
            differential_weight[i], 1
        )

        # torch evaluates all three crossovers under a constant where-condition
        # and keeps only the matching one; this static if/elif traces exactly
        # the same result with fewer ops (cross_strategy is a Python int).
        cross_strategy = STRATEGIES[i][3]
        if cross_strategy == 0:
            key, k_cross = random.split(key)
            trial_vec = DE_binary_crossover(
                k_cross, mutation_vec, current_vec, cross_probability[i]
            )
        elif cross_strategy == 1:
            key, k_cross = random.split(key)
            trial_vec = DE_exponential_crossover(
                k_cross, mutation_vec, current_vec, cross_probability[i]
            )
        else:  # cross_strategy == 2 (arithmetic recombination, no rng)
            trial_vec = DE_arithmetic_recombination(
                mutation_vec, current_vec, cross_probability[i]
            )
        trials.append(trial_vec)

    trial_vectors = etl.stack(trials, axis=0)  # (3, pop_size, dim)
    trial_vectors = clamp(
        enp.reshape(trial_vectors, (3 * pop_size, dim)), lb, ub
    )
    state = CoDEState(
        best_index=state.best_index,
        pop=state.pop,
        fit=state.fit,
        trial_vectors=trial_vectors,
        key=key,
    )
    return trial_vectors, state


def tell(config: CoDE, state: CoDEState, fitness: SymbolicTensor) -> CoDEState:
    """Select the best of the three trial strategies per individual and update
    the population/fitness with torch's ``<=`` replacement rule (key unchanged)."""
    pop_size = config.pop_size
    indices = enp.reshape(enp.arange(3 * pop_size, dtype="int32"), (3, pop_size))
    trans_fit = etl.gather(fitness, indices, axis=0)  # (3, pop_size)
    min_indices = etl.argmin(trans_fit, axis=0)  # (pop_size,)
    min_indices_global = enp.reshape(
        _take_along_axis(indices, enp.expand_dims(min_indices, 0), axis=0),
        (pop_size,),
    )
    trial_fitness_select = etl.gather(fitness, min_indices_global, axis=0)
    trial_vectors_select = etl.gather(
        state.trial_vectors, min_indices_global, axis=0
    )
    compare = trial_fitness_select <= state.fit
    pop = etl.select(enp.expand_dims(compare, 1), trial_vectors_select, state.pop)
    fit = etl.select(compare, trial_fitness_select, state.fit)
    best_index = etl.argmin(fit)
    return CoDEState(
        best_index=best_index,
        pop=pop,
        fit=fit,
        trial_vectors=state.trial_vectors,
        key=state.key,
    )
