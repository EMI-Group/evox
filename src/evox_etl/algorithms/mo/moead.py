"""Functional ETL port of the torch evox MOEA/D algorithm.

Plain functions (no ``@etl.defn`` — ETL has no eager mode; call them only
inside an active trace via ``etl.build``/``etl.run``). 1:1 port of
``src/evox/algorithms/mo/moead.py`` (read-only torch reference): same math,
same draw order, key-first RNG (operator shims). The torch ``step``'s
sequential per-individual update loop is traced with ``etl.while_loop``;
the per-generation parents/crossover/mutation batch is computed up-front
(the batched pairing is equivalent to the per-i torch calls).
"""

import math
from dataclasses import dataclass, replace
from typing import Any, Callable, Optional, Tuple

import numpy as np

import etl
import etl.numpy as enp
import etl.random as random
from etl import core

from evox_etl.algorithms._jit_fix_operator import _take_along_axis, clamp, minimum
from evox_etl.operators.crossover import simulated_binary_half
from evox_etl.operators.mutation import polynomial_mutation
from evox_etl.operators.sampling import uniform_sampling


@dataclass(frozen=True)
class MOEADConfig:
    """MOEA/D hyperparameters (mirrors torch ``MOEAD.__init__`` minus device).

    ``selection_op`` is accepted for signature parity with the torch class
    but is never used by the algorithm (torch ignores it too).
    """

    pop_size: int
    n_objs: int
    lb: np.ndarray
    ub: np.ndarray
    selection_op: Optional[Callable] = None
    mutation_op: Optional[Callable] = None
    crossover_op: Optional[Callable] = None


def _config_flatten(config: MOEADConfig):
    """Zero-child flattening: the config travels as one opaque static node."""
    return [], config


def _config_unflatten(config: MOEADConfig, _children) -> MOEADConfig:
    return config


# ETL v1 rejects numpy arrays as static pytree leaves (they are neither
# TensorSpecs nor static Python values), so the config (which holds lb/ub as
# ndarrays) is registered as a childless pytree node carrying the whole
# config as its context — it then passes through etl.build/etl.run untouched.
etl.register_pytree_node(MOEADConfig, _config_flatten, _config_unflatten)


@dataclass(frozen=True)
class MOEADState:
    """MOEA/D search state; every leaf is an ETL tensor."""

    pop: Any
    fit: Any
    z: Any
    w: Any
    neighbors: Any
    next_generation: Any
    next_parents: Any
    key: Any


def pbi(f: Any, w: Any, z: Any) -> Any:
    """Penalty-based boundary intersection (PBI) scalarizing function.

    1:1 port of the torch ``pbi`` in ``evox.algorithms.mo.moead``.
    """
    norm_w = etl.norm(w, axis=1)
    f = f - z
    d1 = etl.sum(f * w, axes=1) / norm_w
    d2 = etl.norm(f - enp.expand_dims(d1, 1) * w / enp.expand_dims(norm_w, 1), axis=1)
    return d1 + 5.0 * d2


def _bounds(config: MOEADConfig) -> Tuple[Any, Any]:
    """Bake the (dim,) lower/upper bound arrays as graph constants."""
    lb = etl.ops.constant(core.tensor(np.asarray(config.lb, dtype=np.float32)))
    ub = etl.ops.constant(core.tensor(np.asarray(config.ub, dtype=np.float32)))
    return lb, ub


def init(config: MOEADConfig, key: Any) -> MOEADState:
    """Draw the initial MOEA/D state (torch ``MOEAD.__init__`` 1:1).

    ``uniform_sampling`` overwrites the requested pop_size: the effective
    population is the Das-Dennis count ``n_w`` and ``n_neighbor`` is
    ``ceil(n_w / 10)`` (torch does the same).
    """
    w, n_w = uniform_sampling(config.pop_size, config.n_objs)
    n_neighbor = int(math.ceil(n_w / 10))
    dim = config.lb.shape[0]
    lb, ub = _bounds(config)

    population = random.uniform(key, (n_w, dim), 0.0, 1.0, "float32") * (ub - lb) + lb
    fit = enp.full((n_w, config.n_objs), float("inf"), dtype="float32")
    z = enp.zeros((config.n_objs,), dtype="float32")

    # Pairwise squared distances of the weight vectors (no etl.cdist).
    w2 = etl.sum(w * w, axes=1)
    d2 = (
        enp.expand_dims(w2, 1)
        + enp.expand_dims(w2, 0)
        - 2.0 * etl.dot(w, etl.transpose(w, axes=(1, 0)))
    )
    dist = etl.sqrt(enp.maximum(d2, 0.0))
    neighbors = etl.cast(etl.argsort(dist, axis=1, stable=True)[:, :n_neighbor], etl.int32)

    return MOEADState(
        pop=population,
        fit=fit,
        z=z,
        w=w,
        neighbors=neighbors,
        next_generation=enp.zeros((n_w, dim), dtype="float32"),
        next_parents=enp.zeros((n_w, n_neighbor), dtype="int32"),
        key=key,
    )


def init_ask(config: MOEADConfig, state: MOEADState) -> Tuple[Any, MOEADState]:
    """Generation 0 evaluates the whole population (no RNG draw)."""
    return state.pop, state


def init_tell(config: MOEADConfig, state: MOEADState, fitness: Any) -> MOEADState:
    """Record the initial fitness and ideal point (torch ``init_step``)."""
    return replace(state, fit=fitness, z=etl.min(fitness, axes=0))


def ask(config: MOEADConfig, state: MOEADState) -> Tuple[Any, MOEADState]:
    """Produce one offspring per weight vector (torch ``step`` sampling).

    Draws per-row permutations of the neighbor lists, pairs rows i and i+n_w
    for crossover (equivalent to the torch per-i single-pair call), mutates
    and clamps. Returns the offspring batch (n_w, dim).
    """
    key, k_perm, k_cross, k_mut = random.split_n(state.key, 4)
    n_w, n_neighbor = state.next_parents.shape

    perm = etl.argsort(
        random.uniform(k_perm, (n_w, n_neighbor), 0.0, 1.0, "float32"),
        axis=1,
        stable=True,
    )
    next_parents = _take_along_axis(state.neighbors, perm, axis=1)
    parents = next_parents[:, :2]

    x_pairs = etl.concatenate(
        [
            etl.gather(state.pop, parents[:, 0], axis=0),
            etl.gather(state.pop, parents[:, 1], axis=0),
        ],
        axis=0,
    )
    crossover = (
        simulated_binary_half if config.crossover_op is None else config.crossover_op
    )
    crossovered = crossover(k_cross, x_pairs)

    mutation = polynomial_mutation if config.mutation_op is None else config.mutation_op
    lb, ub = _bounds(config)
    offspring = mutation(k_mut, crossovered, lb, ub)
    offspring = clamp(offspring, lb, ub)

    return offspring, replace(
        state, key=key, next_generation=offspring, next_parents=next_parents
    )


def tell(config: MOEADConfig, state: MOEADState, fitness: Any) -> MOEADState:
    """Update z and the neighbor subpopulations (torch ``step`` 1:1).

    The per-i update loop runs as one traced ``etl.while_loop``; ``z`` is a
    loop carry lowered per-i BEFORE the g comparisons, exactly like torch's
    ``self.z = minimum(self.z, off_fit)`` (a pre-loop batch min would
    contaminate the g_old/g_new decisions with future offspring minima).
    """
    n_w = state.next_parents.shape[0]
    n_neighbor = state.next_parents.shape[1]
    z = state.z

    def cond(carry: Tuple[Any, Any, Any, Any]) -> Any:
        i, pop, fit, z = carry
        return i < n_w

    def body(carry: Tuple[Any, Any, Any, Any]) -> Tuple[Any, Any, Any, Any]:
        i, pop, fit, z = carry
        parents = etl.gather(state.next_parents, enp.expand_dims(i, 0), axis=0)[0]
        off = etl.gather(state.next_generation, enp.expand_dims(i, 0), axis=0)[0]
        off_fit = etl.gather(fitness, enp.expand_dims(i, 0), axis=0)[0]
        z = minimum(z, off_fit)

        w_p = etl.gather(state.w, parents, axis=0)
        fit_p = etl.gather(fit, parents, axis=0)
        g_old = pbi(fit_p, w_p, z)
        g_new = pbi(enp.expand_dims(off_fit, 0), w_p, z)
        mask = enp.expand_dims(g_old >= g_new, 1)

        # Replacement scatter is safe: the parents of one row are distinct.
        pop = etl.scatter(
            pop,
            parents,
            etl.select(
                mask,
                etl.tile(enp.expand_dims(off, 0), (n_neighbor, 1)),
                etl.gather(pop, parents, axis=0),
            ),
            axis=0,
        )
        fit = etl.scatter(
            fit,
            parents,
            etl.select(
                mask,
                etl.tile(enp.expand_dims(off_fit, 0), (n_neighbor, 1)),
                etl.gather(fit, parents, axis=0),
            ),
            axis=0,
        )
        i = etl.cast(i + 1, etl.int32)
        return i, pop, fit, z

    i0 = enp.zeros((), dtype="int32")
    i, pop, fit, z = etl.while_loop(cond, body, (i0, state.pop, state.fit, z))
    return replace(state, pop=pop, fit=fit, z=z)
