"""Functional port of the torch RVEAa algorithm (``src/evox/algorithms/mo/rveaa.py``).

Plain functions only (no ``@etl.defn``, per DESIGN.md §4.3) — the workflow traces
``init``/``init_ask``/``init_tell``/``ask``/``tell`` via ``etl.build``/``etl.run``.

Port notes:
- The torch OOP ``step()`` is split at its ``self.evaluate(offspring)`` call:
  ``ask`` returns the offspring batch, ``tell`` runs the RVEAa selection.
  ``tell(config, state, fitness)`` needs the offspring for the population merge,
  so ``ask`` stores it in the state (``state.offspring``) — the same
  candidates-in-state pattern used by the DE ports.
- torch's eager-if / ``torch.cond`` dual paths of ``_update_pop_and_rv`` become
  ONE ``etl.select``-based path (both branches are pure and cheap).
- The effective population size is the Das-Dennis count ``n_v`` from
  ``uniform_sampling`` (torch overwrites ``self.pop_size`` with it); the SBX
  quirk of torch ``simulated_binary`` (last odd row dropped) is preserved 1:1.
"""

from dataclasses import dataclass, replace
from typing import Any, Optional, Tuple

import numpy as np

import etl
import etl.numpy as enp
import etl.random as random

from evox_etl.algorithms._operator_shims import (
    clamp,
    nanmax,
    nanmin,
    non_dominate_rank,
    polynomial_mutation,
    randint,
    ref_vec_guided,
    simulated_binary,
    uniform_sampling,
)

Tensor = etl.SymbolicTensor


@dataclass(frozen=True, eq=False)
class RVEAaConfig:
    """Config mirroring torch ``RVEAa.__init__`` (``device`` dropped)."""

    pop_size: int
    n_objs: int
    lb: np.ndarray
    ub: np.ndarray
    alpha: float = 2.0
    fr: float = 0.1
    max_gen: int = 100
    selection_op: Optional[Any] = None
    mutation_op: Optional[Any] = None
    crossover_op: Optional[Any] = None

    def __post_init__(self):
        assert self.lb.shape == self.ub.shape and self.lb.ndim == 1 and self.ub.ndim == 1
        assert self.lb.dtype == self.ub.dtype


@dataclass(frozen=True, eq=False)
class RVEAaState:
    """Tensors mirroring the torch RVEAa Mutable attributes plus the RNG key.

    ``offspring`` carries the last ``ask`` batch so ``tell(config, state,
    fitness)`` can merge it with the population (torch ``step`` keeps it as a
    local variable between the evaluate call and the merge).
    """

    pop: Tensor
    fit: Tensor
    reference_vector: Tensor
    init_v: Tensor
    gen: Tensor
    rv_adapt_every: Tensor
    offspring: Tensor
    key: Tensor


def _bounds(config: RVEAaConfig) -> Tuple[Tensor, Tensor, Tensor]:
    """Bake the lb/ub config arrays as (dim,) float32 graph constants plus the
    (2, dim) boundary stack expected by the polynomial-mutation shim."""
    lb = etl.ops.constant(etl.core.tensor(np.asarray(config.lb, dtype=np.float32)))
    ub = etl.ops.constant(etl.core.tensor(np.asarray(config.ub, dtype=np.float32)))
    boundary = enp.stack([lb, ub], axis=0)
    return lb, ub, boundary


def init(config: RVEAaConfig, key: Tensor) -> RVEAaState:
    """Draw the initial state (torch ``RVEAa.__init__``): uniform population,
    inf fitness, and the 2*n_v reference vectors (Das-Dennis half + random
    half) with the Das-Dennis points kept as ``init_v``."""
    key, k_pop, k_v1 = random.split_n(key, 3)
    dim = config.lb.shape[0]
    n_objs = config.n_objs
    lb, ub, _ = _bounds(config)

    v, n_v = uniform_sampling(config.pop_size, n_objs)
    population = random.uniform(k_pop, (n_v, dim), 0.0, 1.0, "float32") * (ub - lb) + lb
    fit = enp.full((n_v, n_objs), float("inf"), dtype="float32")
    v1 = random.uniform(k_v1, (n_v, n_objs), 0.0, 1.0, "float32")
    reference_vector = etl.concatenate([v, v1], axis=0)
    gen = enp.zeros((), dtype="int32")
    rv_adapt_every = enp.full(
        (), float(max(round(1.0 / config.fr), 1.0)), dtype="float32"
    )
    return RVEAaState(
        pop=population,
        fit=fit,
        reference_vector=reference_vector,
        init_v=v,
        gen=gen,
        rv_adapt_every=rv_adapt_every,
        offspring=population,
        key=key,
    )


def init_ask(config: RVEAaConfig, state: RVEAaState) -> Tuple[Tensor, RVEAaState]:
    """Generation 0 evaluates the whole initial population (no new draws)."""
    return state.pop, state


def init_tell(config: RVEAaConfig, state: RVEAaState, fitness: Tensor) -> RVEAaState:
    """Store the initial fitness (torch ``init_step``: ``fit = evaluate(pop)``)."""
    return replace(state, fit=fitness)


def ask(config: RVEAaConfig, state: RVEAaState) -> Tuple[Tensor, RVEAaState]:
    """Produce the offspring batch (torch ``RVEAa.step`` up to the evaluate):
    mating pool over the non-all-NaN rows, SBX, polynomial mutation, clamp."""
    key, k_mate, k_cross, k_mut = random.split_n(state.key, 4)
    gen = etl.cast(state.gen + 1, etl.int32)
    lb, ub, boundary = _bounds(config)
    pop = state.pop
    pop_rows = pop.shape[0]
    # Fixed effective pop size (Das-Dennis count, torch self.pop_size): pop
    # grows to 2*n_v rows after the first tell, but the mating pool always
    # draws n_v candidates.
    n_v = state.reference_vector.shape[0] // 2

    # torch _mating_pool: valid rows sort to the front (NaN rows get the int32
    # max sentinel), then n_v random draws over the valid prefix.
    valid_mask = enp.logical_not(etl.ops.reduce_max(etl.isnan(pop), axes=1))
    num_valid = etl.cast(etl.sum(etl.cast(valid_mask, etl.int32)), etl.int32)
    mating_pool = randint(k_mate, 0, num_valid, (n_v,), dtype=etl.int32)
    sorted_indices = etl.argsort(
        etl.cast(
            etl.select(valid_mask, enp.arange(pop_rows, dtype="int32"), 2147483647),
            etl.int32,
        ),
        axis=0,
        stable=True,
    )
    pool = etl.gather(pop, sorted_indices, axis=0)
    mated = etl.gather(pool, mating_pool, axis=0)

    crossovered = simulated_binary(k_cross, mated)
    offspring = polynomial_mutation(k_mut, crossovered, boundary)
    offspring = clamp(offspring, lb, ub)
    return offspring, replace(state, gen=gen, offspring=offspring, key=key)


def tell(config: RVEAaConfig, state: RVEAaState, fitness: Tensor) -> RVEAaState:
    """RVEAa selection (torch ``RVEAa.step`` after the evaluate): merge,
    non-dominated rank, ref-vector-guided survivors, then reference-vector
    regeneration + adaptation and final batch truncation."""
    pop, fit = state.pop, state.fit
    gen = state.gen
    n_objs = config.n_objs
    # Fixed effective pop size (Das-Dennis count): the reference vector keeps
    # (2*n_v, n_objs) throughout, so n_v = half its rows.
    n_v = state.reference_vector.shape[0] // 2

    merge_pop = etl.concatenate([pop, state.offspring], axis=0)
    merge_fit = etl.concatenate([fit, fitness], axis=0)

    rank = non_dominate_rank(merge_fit)
    merge_fit = etl.select(enp.expand_dims(rank, 1) == 0, merge_fit, float("nan"))
    merge_pop = etl.select(enp.expand_dims(rank, 1) == 0, merge_pop, float("nan"))

    theta = (etl.cast(gen, etl.float32) / config.max_gen) ** config.alpha
    survivor, survivor_fit = ref_vec_guided(
        merge_pop, merge_fit, state.reference_vector, theta
    )

    # --- torch _rv_regeneration over reference_vector[n_v:] ---
    key = state.key
    key, k_reg = random.split(key)
    v = state.reference_vector[n_v:]
    pop_obj = survivor_fit - nanmin(survivor_fit, dim=0)[0]
    # Row-normalized dot product == torch F.cosine_similarity (incl. NaN rows).
    an = pop_obj / etl.norm(pop_obj, axis=1, keepdims=True)
    bn = v / etl.norm(v, axis=1, keepdims=True)
    cosine = etl.dot(an, etl.transpose(bn, axes=(1, 0)))  # (2*n_v, n_v)
    mask = etl.isnan(cosine)
    input_tensor = etl.select(mask, float("-inf"), cosine)
    associate = etl.argmax(input_tensor, axis=1)
    associate = etl.cast(
        etl.select(input_tensor[:, 0] == float("-inf"), -1, associate), etl.int32
    )
    invalid = etl.cast(
        etl.sum(
            etl.cast(
                etl.equal(
                    enp.expand_dims(associate, 1),
                    enp.expand_dims(enp.arange(n_v, dtype="int32"), 0),
                ),
                etl.int32,
            ),
            axes=0,
        ),
        etl.int32,
    )
    rand = random.uniform(k_reg, (n_v, n_objs), 0.0, 1.0, "float32") * nanmax(
        pop_obj, dim=0
    )[0]
    new_v = etl.select(enp.expand_dims(invalid == 0, 1), rand, v)

    # --- torch _rv_adaptation vs _no_rv_adaptation: one select-based path ---
    adapted = state.init_v * (
        nanmax(survivor_fit, dim=0)[0] - nanmin(survivor_fit, dim=0)[0]
    )
    kept = state.reference_vector[:n_v]
    v_adapt = etl.select(
        etl.remainder(gen, etl.cast(state.rv_adapt_every, etl.int32)) == 0,
        adapted,
        kept,
    )

    # --- torch _batch_truncation vs _no_batch_truncation: one select path ---
    n_total = 2 * n_v
    obj = survivor_fit
    an = obj / etl.norm(obj, axis=1, keepdims=True)
    cosine = etl.dot(an, etl.transpose(an, axes=(1, 0)))  # (n_total, n_total)
    not_all_nan = enp.logical_not(etl.ops.reduce_max(etl.isnan(cosine), axes=1))
    mask = enp.logical_and(
        etl.eye(n_total, dtype=etl.bool_), enp.expand_dims(not_all_nan, 1)
    )
    cosine = etl.select(mask, 0.0, cosine)
    sorted_values = etl.sort(-cosine, axis=1)
    first_col = sorted_values[:, 0]
    first_col = etl.select(etl.isnan(first_col), float("-inf"), first_col)
    _rank = etl.argsort(first_col, axis=0)  # torch computes this but never uses it
    keep_mask = enp.arange(n_total, dtype="int32") >= n_v
    trunc_pop = etl.select(enp.expand_dims(keep_mask, 1), survivor, float("nan"))
    trunc_fit = etl.select(enp.expand_dims(keep_mask, 1), survivor_fit, float("nan"))
    pop_out = etl.select(gen == config.max_gen, trunc_pop, survivor)
    fit_out = etl.select(gen == config.max_gen, trunc_fit, survivor_fit)

    reference_vector = etl.concatenate([v_adapt, new_v], axis=0)
    return replace(
        state,
        pop=pop_out,
        fit=fit_out,
        reference_vector=reference_vector,
        key=key,
    )
