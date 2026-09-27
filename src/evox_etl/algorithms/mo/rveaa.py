"""Functional port of the torch RVEAa algorithm (``src/evox/algorithms/mo/rveaa.py``).

Plain functions only (no ``@etl.defn``, per DESIGN.md §4.3) following the step
protocol of ``evox_etl.core.algorithm``: the workflow traces ``init``/
``init_step``/``step`` via ``etl.build``/``etl.run``.

Port notes:
- The torch OOP ``step()`` is one fused function here: the offspring phase
  (mating pool -> SBX -> polynomial mutation), ``fitness = evaluate(offspring)``
  through the workflow-owned opaque closure, then the RVEAa selection. The
  selection phase needs the offspring for the population merge, so the
  offspring phase stores it in the intermediate state (``state.offspring``) —
  the same candidates-in-state pattern used by the DE ports.
- torch's eager-if / ``torch.cond`` dual paths of ``_update_pop_and_rv`` become
  ONE ``etl.select``-based path (both branches are pure and cheap).
- The effective population size is the Das-Dennis count ``n_v`` from
  ``uniform_sampling`` (torch overwrites ``self.pop_size`` with it); the SBX
  quirk of torch ``simulated_binary`` (last odd row dropped) is preserved 1:1.
"""

from dataclasses import dataclass, replace
from typing import Callable, Optional

import etl
import etl.numpy as enp
import etl.random as random

from evox_etl.algorithms._config_utils import ArrayLike, bake_bounds, normalize_bounds
from evox_etl.operators.jit_fix_operator import clamp, nanmax, nanmin, randint
from evox_etl.operators.crossover import simulated_binary
from evox_etl.operators.mutation import polynomial_mutation
from evox_etl.operators.sampling import uniform_sampling
from evox_etl.operators.selection import non_dominate_rank, ref_vec_guided

Tensor = etl.SymbolicTensor


@dataclass(frozen=True)
class RVEAaConfig:
    """Config mirroring torch ``RVEAa.__init__`` (``device`` dropped).

    Signature parity: the torch RVEAa constructor's optional
    ``selection_op``/``mutation_op``/``crossover_op`` are honored here too.
    A ``None`` op field means the torch default (``ref_vec_guided`` /
    ``simulated_binary`` / ``polynomial_mutation``). Custom signatures:
    ``selection_op(x, f, v, theta)`` (no key), ``crossover_op(key, x)`` and
    ``mutation_op(key, x, lb, ub)``. ``lb``/``ub`` are stored as flat float
    tuples of length ``dim`` (see `make_rveaa` for the array-accepting
    constructor).
    """

    pop_size: int
    n_objs: int
    lb: tuple[float, ...]
    ub: tuple[float, ...]
    alpha: float = 2.0
    fr: float = 0.1
    max_gen: int = 100
    selection_op: Optional[Callable] = None
    mutation_op: Optional[Callable] = None
    crossover_op: Optional[Callable] = None


def _config_flatten(config: RVEAaConfig):
    """Zero-child flattening: the config travels as one opaque static node."""
    return [], config


def _config_unflatten(config: RVEAaConfig, _children) -> RVEAaConfig:
    return config


# This registration is REQUIRED: the config carries optional callable op
# fields (selection_op/mutation_op/crossover_op), and functions are not valid
# static pytree leaves — so the config travels as one opaque childless node
# through etl.build/etl.run untouched.
etl.register_pytree_node(RVEAaConfig, _config_flatten, _config_unflatten)


def make_rveaa(
    pop_size: int,
    n_objs: int,
    lb: ArrayLike,
    ub: ArrayLike,
    alpha: float = 2.0,
    fr: float = 0.1,
    max_gen: int = 100,
    selection_op: Optional[Callable] = None,
    mutation_op: Optional[Callable] = None,
    crossover_op: Optional[Callable] = None,
) -> RVEAaConfig:
    """Construct an RVEAaConfig with ``lb``/``ub`` normalized to flat float tuples.

    Raises ValueError (not the old AssertionError) when a bound is not 1-D or
    the two shapes differ, or when an op field is not callable (and not None).
    The old ``__post_init__`` dtype-equality check is gone: with tuple storage
    per-side dtype is vacuous — each bound is dtype-preservingly rounded to
    plain floats and re-cast to float32 at bake time anyway. A ``None`` op field
    means the algorithm default (``ref_vec_guided`` / ``simulated_binary`` /
    ``polynomial_mutation``).
    """
    for name, op in (
        ("selection_op", selection_op),
        ("mutation_op", mutation_op),
        ("crossover_op", crossover_op),
    ):
        if op is not None and not callable(op):
            raise ValueError(f"{name} must be callable or None, got {op!r}")
    lb, ub = normalize_bounds(lb, ub)
    return RVEAaConfig(
        pop_size=pop_size,
        n_objs=n_objs,
        lb=lb,
        ub=ub,
        alpha=alpha,
        fr=fr,
        max_gen=max_gen,
        selection_op=selection_op,
        mutation_op=mutation_op,
        crossover_op=crossover_op,
    )


@dataclass(frozen=True, eq=False)
class RVEAaState:
    """Tensors mirroring the torch RVEAa Mutable attributes plus the RNG key.

    ``offspring`` carries the offspring batch from the generation phase of
    ``step`` into its selection phase (torch ``step`` keeps it as a local
    variable between the evaluate call and the merge).
    """

    pop: Tensor
    fit: Tensor
    reference_vector: Tensor
    init_v: Tensor
    gen: Tensor
    rv_adapt_every: Tensor
    offspring: Tensor
    key: Tensor


def init(config: RVEAaConfig, key: Tensor) -> RVEAaState:
    """Draw the initial state (torch ``RVEAa.__init__``): uniform population,
    inf fitness, and the 2*n_v reference vectors (Das-Dennis half + random
    half) with the Das-Dennis points kept as ``init_v``."""
    key, k_pop, k_v1 = random.split_n(key, 3)
    dim = len(config.lb)
    n_objs = config.n_objs
    lb, ub = bake_bounds(config.lb, config.ub)

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


def init_step(
    config: RVEAaConfig, state: RVEAaState, evaluate: Callable[[Tensor], Tensor]
) -> RVEAaState:
    """Generation 0 (fused torch ``init_step``): evaluate the whole initial
    population and store its fitness (no new draws)."""
    fitness = evaluate(state.pop)
    return replace(state, fit=fitness)


def step(
    config: RVEAaConfig, state: RVEAaState, evaluate: Callable[[Tensor], Tensor]
) -> RVEAaState:
    """Run ONE full RVEAa generation (fused torch ``RVEAa.step``).

    Phase 1 (offspring generation): mating pool over the non-all-NaN rows, SBX,
    polynomial mutation, clamp — offspring stored in the intermediate state.
    Phase 2: ``fitness = evaluate(offspring)`` through the workflow-owned
    opaque closure. Phase 3 (selection): merge, non-dominated rank,
    ref-vector-guided survivors, then reference-vector regeneration +
    adaptation and final batch truncation.
    """
    key, k_mate, k_cross, k_mut = random.split_n(state.key, 4)
    gen = etl.cast(state.gen + 1, etl.int32)
    lb, ub = bake_bounds(config.lb, config.ub)
    pop = state.pop
    pop_rows = pop.shape[0]
    # Fixed effective pop size (Das-Dennis count, torch self.pop_size): pop
    # grows to 2*n_v rows after the first generation, but the mating pool
    # always draws n_v candidates.
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

    crossover_fn = (
        config.crossover_op if config.crossover_op is not None else simulated_binary
    )
    crossovered = crossover_fn(k_cross, mated)
    mutation_fn = (
        config.mutation_op if config.mutation_op is not None else polynomial_mutation
    )
    offspring = mutation_fn(k_mut, crossovered, lb, ub)
    offspring = clamp(offspring, lb, ub)
    state = replace(state, gen=gen, offspring=offspring, key=key)

    fitness = evaluate(offspring)

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
    selection_fn = (
        config.selection_op if config.selection_op is not None else ref_vec_guided
    )
    survivor, survivor_fit = selection_fn(
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
