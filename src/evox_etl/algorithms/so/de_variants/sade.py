"""Functional port of torch evox SaDE
(``src/evox/algorithms/so/de_variants/sade.py``, read-only reference).

Port notes:
- The torch OOP ``step()`` is split at its ``self.evaluate(trial_vector)`` call:
  everything before the evaluate lives in ``ask`` (returns the trial vectors),
  everything after lives in ``tell`` (selection + memory updates).
- Stateless etl RNG: the key is stored in the state and advanced in ``ask``
  with ONE ``random.split`` per torch random draw, in torch's draw order.
  ``tell`` never draws randomness.
- ``lb``/``ub`` config fields are normalized to tuples of plain Python floats
  (np.ndarray fields are rejected by ``etl.build``); they are baked as graph
  constants inside the functions.
- torch's per-i scatter-add loop updating success/failure memory is replaced by
  its exact vectorized equivalent: roll + zero row 0, then write the
  per-strategy success/failure counts of this generation into row 0.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

import etl
import etl.numpy as enp
import etl.random as random

from evox_etl.algorithms._operator_shims import (
    DE_arithmetic_recombination,
    DE_binary_crossover,
    DE_differential_sum,
    DE_exponential_crossover,
    clamp,
    select_rand_pbest,
)
from evox_etl.algorithms._shim_selection_basic import _take_along_axis

Tensor = etl.SymbolicTensor

# Strategy codes (4 bits): [base_vec_prim, base_vec_sec, diff_num, cross_strategy]
# base_vec: 0="rand", 1="best", 2="pbest", 3="current"; cross: 0=bin, 1=exp, 2=arith
STRATEGY_POOL = ((0, 0, 1, 0), (0, 1, 2, 0), (0, 0, 2, 0), (0, 0, 1, 2))


@dataclass(frozen=True)
class SaDE:
    """Config mirroring torch ``SaDE.__init__`` (device dropped)."""

    pop_size: int
    lb: Any
    ub: Any
    diff_padding_num: int = 9
    LP: int = 50

    def __post_init__(self):
        assert self.pop_size >= 9
        assert len(self.lb) == len(self.ub)
        object.__setattr__(self, "lb", tuple(float(v) for v in self.lb))
        object.__setattr__(self, "ub", tuple(float(v) for v in self.ub))


@dataclass(frozen=True)
class SaDEState:
    """Tensors mirroring the torch SaDE Mutable attributes plus the RNG key."""

    gen_iter: Tensor
    best_index: Tensor
    Memory_FCR: Tensor
    pop: Tensor
    fit: Tensor
    success_memory: Tensor
    failure_memory: Tensor
    CR_memory: Tensor
    trial_vectors: Tensor
    strategy_ids: Tensor
    CRs_vec: Tensor
    key: Tensor


def _bake_lb_ub(config: SaDE) -> tuple[Tensor, Tensor]:
    """Bake the normalized lb/ub tuples as (1, dim) float32 graph constants."""
    lb = etl.ops.constant(etl.core.tensor(np.asarray(config.lb, dtype=np.float32)))
    ub = etl.ops.constant(etl.core.tensor(np.asarray(config.ub, dtype=np.float32)))
    return enp.reshape(lb, (1, -1)), enp.reshape(ub, (1, -1))


def init(config: SaDE, key: Tensor) -> SaDEState:
    """Draw the initial state: randn-scaled population (torch quirk — no
    uniform, no clamp), inf fitness, zeroed counters and NaN CR memory."""
    key, subkey = random.split(key)
    pop_size, dim = config.pop_size, len(config.lb)
    lb, ub = _bake_lb_ub(config)

    gen_iter = enp.full((), 0, dtype="int64")
    best_index = enp.full((), 0, dtype="int64")
    Memory_FCR = enp.full((2, 100), 0.5, dtype="float32")
    pop = random.normal(subkey, (pop_size, dim), 0.0, 1.0, "float32") * (ub - lb) + lb
    fit = enp.full((pop_size,), float("inf"), dtype="float32")
    success_memory = enp.zeros((config.LP, 4), dtype="int32")
    failure_memory = enp.zeros((config.LP, 4), dtype="int32")
    CR_memory = enp.full((config.LP, 4), float("nan"), dtype="float32")
    strategy_ids = enp.zeros((pop_size,), dtype="int32")
    CRs_vec = enp.zeros((pop_size,), dtype="float32")

    return SaDEState(
        gen_iter=gen_iter,
        best_index=best_index,
        Memory_FCR=Memory_FCR,
        pop=pop,
        fit=fit,
        success_memory=success_memory,
        failure_memory=failure_memory,
        CR_memory=CR_memory,
        trial_vectors=pop,
        strategy_ids=strategy_ids,
        CRs_vec=CRs_vec,
        key=key,
    )


def ask(config: SaDE, state: SaDEState) -> tuple[Tensor, SaDEState]:
    """Generate trial vectors (torch ``SaDE.step`` up to ``self.evaluate``):
    adaptive strategy probabilities, per-individual F/CR, DE mutation and
    crossover."""
    pop_size, dim = config.pop_size, len(config.lb)
    lb, ub = _bake_lb_ub(config)
    strategy_pool = etl.ops.constant(
        etl.core.tensor(np.asarray(STRATEGY_POOL, dtype=np.int32))
    )

    # One split per torch random draw, in torch's draw order.
    key = state.key
    key, k_mult = random.split(key)  # strategy_ids multinomial
    key, k_cr = random.split(key)  # CRs_vec randn
    key, k_cr_rep = random.split(key)  # CRs_vec_repair randn
    key, k_f = random.split(key)  # differential_weight randn
    key, k_diff = random.split(key)  # DE_differential_sum randint
    key, k_pbest = random.split(key)  # select_rand_pbest randint
    key, k_bin = random.split(key)  # binary crossover (mask + rind)
    key, k_exp = random.split(key)  # exponential crossover (nn + ll)

    success_sum = etl.cast(etl.sum(state.success_memory, axes=0), "float32")
    failure_sum = etl.cast(etl.sum(state.failure_memory, axes=0), "float32")
    S_mat = success_sum / (success_sum + failure_sum) + 0.01
    strategy_p_update = S_mat / etl.sum(S_mat, axes=0)
    strategy_p_init = enp.full((4,), 0.25, dtype="float32")
    strategy_p = etl.select(
        state.gen_iter >= config.LP, strategy_p_update, strategy_p_init
    )

    CRM_init = enp.full((4,), 0.5, dtype="float32")
    CRM_update = etl.median(state.CR_memory, axis=0)
    CRM = etl.select(state.gen_iter > config.LP, CRM_update, CRM_init)

    strategy_ids = random.multinomial(k_mult, strategy_p, pop_size)

    CRs_vec = random.normal(k_cr, (pop_size, 4), 0.0, 1.0, "float32") * 0.1 + CRM
    CRs_vec_repair = (
        random.normal(k_cr_rep, (pop_size, 4), 0.0, 1.0, "float32") * 0.1 + CRM
    )
    mask = etl.logical_or(CRs_vec < 0, CRs_vec > 1)
    CRs_vec = etl.select(mask, CRs_vec_repair, CRs_vec)

    differential_weight = random.normal(k_f, (pop_size,), 0.0, 1.0, "float32") * 0.3 + 0.5
    cross_probability = enp.reshape(
        _take_along_axis(CRs_vec, enp.expand_dims(strategy_ids, 1), axis=1),
        (pop_size,),
    )

    strategy_code = etl.gather(strategy_pool, strategy_ids, axis=0)
    base_vec_prim_type = strategy_code[:, 0]
    base_vec_sec_type = strategy_code[:, 1]
    num_diff_vectors = strategy_code[:, 2]
    cross_strategy = strategy_code[:, 3]

    difference_sum, rand_vec_idx = DE_differential_sum(
        k_diff,
        config.diff_padding_num,
        num_diff_vectors,
        enp.arange(pop_size, dtype="int32"),
        state.pop,
    )

    rand_vec = etl.gather(state.pop, rand_vec_idx, axis=0)
    best_vec = etl.tile(
        enp.expand_dims(etl.gather(state.pop, state.best_index, axis=0), 0),
        (pop_size, 1),
    )
    pbest_vec = select_rand_pbest(k_pbest, 0.05, state.pop, state.fit)
    current_vec = state.pop
    vector_merge = etl.stack([rand_vec, best_vec, pbest_vec, current_vec], axis=0)

    base_vector_prim = enp.zeros((pop_size, dim), dtype="float32")
    base_vector_sec = enp.zeros((pop_size, dim), dtype="float32")
    for i in range(4):
        base_vector_prim = etl.select(
            enp.expand_dims(base_vec_prim_type == i, 1),
            vector_merge[i],
            base_vector_prim,
        )
        base_vector_sec = etl.select(
            enp.expand_dims(base_vec_sec_type == i, 1),
            vector_merge[i],
            base_vector_sec,
        )

    base_vector = base_vector_prim + enp.expand_dims(differential_weight, 1) * (
        base_vector_sec - base_vector_prim
    )
    mutation_vector = base_vector + difference_sum * enp.expand_dims(
        differential_weight, 1
    )

    # torch evaluates all torch.where args eagerly, so all three crossovers run
    # (and draw randomness) unconditionally — same subkey order here.
    trial_bin = DE_binary_crossover(k_bin, mutation_vector, current_vec, cross_probability)
    trial_exp = DE_exponential_crossover(k_exp, mutation_vector, current_vec, cross_probability)
    trial_arith = DE_arithmetic_recombination(mutation_vector, current_vec, cross_probability)

    trial_vector = enp.zeros((pop_size, dim), dtype="float32")
    trial_vector = etl.select(
        enp.expand_dims(cross_strategy == 0, 1), trial_bin, trial_vector
    )
    trial_vector = etl.select(
        enp.expand_dims(cross_strategy == 1, 1), trial_exp, trial_vector
    )
    trial_vector = etl.select(
        enp.expand_dims(cross_strategy == 2, 1), trial_arith, trial_vector
    )
    trial_vector = clamp(trial_vector, lb, ub)

    CRs_vec_out = enp.reshape(
        _take_along_axis(CRs_vec, enp.expand_dims(strategy_ids, 1), axis=1),
        (pop_size,),
    )

    return trial_vector, SaDEState(
        gen_iter=state.gen_iter,
        best_index=state.best_index,
        Memory_FCR=state.Memory_FCR,
        pop=state.pop,
        fit=state.fit,
        success_memory=state.success_memory,
        failure_memory=state.failure_memory,
        CR_memory=state.CR_memory,
        trial_vectors=trial_vector,
        strategy_ids=strategy_ids,
        CRs_vec=CRs_vec_out,
        key=key,
    )


def tell(config: SaDE, state: SaDEState, fitness: Tensor) -> SaDEState:
    """Apply selection to the trial vectors and update the success/failure/CR
    memories (torch ``SaDE.step`` after ``self.evaluate``)."""
    pop_size = config.pop_size
    LP = config.LP

    gen_iter = state.gen_iter + 1
    compare = fitness <= state.fit
    pop = etl.select(enp.expand_dims(compare, 1), state.trial_vectors, state.pop)
    fit = etl.select(compare, fitness, state.fit)
    best_index = etl.argmin(fit)

    # torch: roll, zero row 0, then per-i scatter-ADD 1 at (0, strategy_ids[i])
    # when compare[i]. Exact vectorized equivalent: after roll+zero, row 0 holds
    # the per-strategy success counts of this generation.
    row0_mask = enp.expand_dims(enp.arange(LP, dtype="int32") == 0, 1)
    onehot = enp.expand_dims(state.strategy_ids, 1) == enp.expand_dims(
        enp.arange(4, dtype="int32"), 0
    )

    success_rolled = etl.roll(state.success_memory, 1, axis=0)
    success_rolled = etl.select(
        row0_mask, enp.zeros((LP, 4), dtype="int32"), success_rolled
    )
    success_counts = etl.sum(
        enp.expand_dims(etl.cast(compare, "int32"), 1) * onehot, axes=0
    )
    success_memory = etl.select(
        row0_mask, enp.expand_dims(success_counts, 0), success_rolled
    )

    failure_rolled = etl.roll(state.failure_memory, 1, axis=0)
    failure_rolled = etl.select(
        row0_mask, enp.zeros((LP, 4), dtype="int32"), failure_rolled
    )
    failure_counts = etl.sum(
        enp.expand_dims(
            enp.ones((pop_size,), dtype="int32") - etl.cast(compare, "int32"), 1
        )
        * onehot,
        axes=0,
    )
    failure_memory = etl.select(
        row0_mask, enp.expand_dims(failure_counts, 0), failure_rolled
    )

    # torch per-i CR-memory loop, ported 1:1 as a static Python loop.
    CR_memory = state.CR_memory
    for i in range(pop_size):
        str_idx = state.strategy_ids[i]
        CR_memory_t = etl.transpose(CR_memory, axes=(1, 0))  # (4, LP)
        CR_mk = enp.reshape(
            etl.gather(CR_memory_t, enp.reshape(str_idx, (1,)), axis=0), (LP,)
        )
        CR_mk_up = etl.roll(CR_mk, 1, axis=0)
        CR_mk_up = etl.select(
            enp.arange(LP, dtype="int32") == 0,
            enp.expand_dims(state.CRs_vec[i], 0),
            CR_mk_up,
        )
        row_mask = enp.expand_dims(enp.arange(4, dtype="int32") == str_idx, 1)
        CR_memory_up = etl.select(row_mask, enp.expand_dims(CR_mk_up, 0), CR_memory_t)
        CR_memory_up = etl.transpose(CR_memory_up, axes=(1, 0))
        CR_memory = etl.select(compare[i], CR_memory_up, CR_memory)

    return SaDEState(
        gen_iter=gen_iter,
        best_index=best_index,
        Memory_FCR=state.Memory_FCR,
        pop=pop,
        fit=fit,
        success_memory=success_memory,
        failure_memory=failure_memory,
        CR_memory=CR_memory,
        trial_vectors=state.trial_vectors,
        strategy_ids=state.strategy_ids,
        CRs_vec=state.CRs_vec,
        key=state.key,
    )
