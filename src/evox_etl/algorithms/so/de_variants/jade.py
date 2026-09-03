"""Functional ETL port of the torch evox ``JaDE`` (adaptive DE) algorithm.

Plain-function port of ``src/evox/algorithms/so/de_variants/jade.py`` (read-only
torch reference), following the ``DESIGN.md`` §4-5 binding: ``init``/``ask``/
``tell`` replace the OOP class, the torch ``step`` is split at its
``self.evaluate`` call (``ask`` produces the trial population and stores it in
``trial_vectors`` together with the per-individual ``F_vec``/``CR_vec`` needed by
the adaptation; ``tell`` performs selection + F_u/CR_u adaptation).

Deviations from torch (mathematically equivalent):
- The RNG key is stored in the state and advanced at every draw (etl RNG is
  stateless); each ``torch.randn``/``torch.rand``/``torch.randint`` draw gets its
  own ``random.split`` subkey, in the same draw order with the same distribution.
- ``lb``/``ub``/``mean``/``stdev`` config fields are normalized to tuples of
  plain Python floats (``etl.build`` rejects np.ndarray static args) and baked as
  graph constants inside each function.
"""

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np

import etl
import etl.numpy as enp
import etl.random as random

from evox_etl.algorithms._operator_shims import clamp, clamp_float, select_rand_pbest

Tensor = etl.SymbolicTensor


@dataclass(frozen=True)
class JaDE:
    """JaDE configuration (mirrors the torch ``JaDE.__init__`` signature)."""

    pop_size: int
    lb: Any  # lower bounds, (dim,) array-like; stored as a tuple of Python floats
    ub: Any  # upper bounds, (dim,) array-like; stored as a tuple of Python floats
    num_difference_vectors: int = 1
    mean: Any = None  # optional normal-init mean; stored as a tuple of Python floats
    stdev: Any = None  # optional normal-init stdev; stored as a tuple of Python floats
    c: float = 0.1

    def __post_init__(self) -> None:
        assert self.pop_size >= 4
        assert len(self.lb) == len(self.ub)
        object.__setattr__(self, "lb", tuple(float(v) for v in self.lb))
        object.__setattr__(self, "ub", tuple(float(v) for v in self.ub))
        if self.mean is not None:
            object.__setattr__(self, "mean", tuple(float(v) for v in self.mean))
        if self.stdev is not None:
            object.__setattr__(self, "stdev", tuple(float(v) for v in self.stdev))


@dataclass(frozen=True)
class JaDEState:
    """JaDE algorithm state (all leaves are etl tensors)."""

    pop: Tensor  # (pop_size, dim) float32
    fit: Tensor  # (pop_size,) float32
    F_u: Tensor  # (pop_size,) float32 adaptive mutation factors
    CR_u: Tensor  # (pop_size,) float32 adaptive crossover rates
    trial_vectors: Tensor  # (pop_size, dim) float32 candidates pending evaluation
    F_vec: Tensor  # (pop_size,) float32 per-individual F from ask (tell adaptation)
    CR_vec: Tensor  # (pop_size,) float32 per-individual CR from ask (tell adaptation)
    key: Tensor  # () int64 rng key


def _bake_bounds(config: JaDE) -> Tuple[Tensor, Tensor]:
    """Bake the (1, dim) lb/ub graph constants from the config tuples."""
    lb = etl.ops.constant(etl.core.tensor(np.asarray(config.lb, dtype=np.float32)))
    ub = etl.ops.constant(etl.core.tensor(np.asarray(config.ub, dtype=np.float32)))
    return enp.reshape(lb, (1, -1)), enp.reshape(ub, (1, -1))


def init(config: JaDE, key: Tensor) -> JaDEState:
    """Create the initial state: uniform/normal population, inf fitness, F_u=CR_u=0.5."""
    pop_size, dim = config.pop_size, len(config.lb)
    lb, ub = _bake_bounds(config)
    key, k_pop = random.split(key)

    if config.mean is not None and config.stdev is not None:
        mean = enp.reshape(
            etl.ops.constant(
                etl.core.tensor(np.asarray(config.mean, dtype=np.float32))
            ),
            (1, -1),
        )
        stdev = enp.reshape(
            etl.ops.constant(
                etl.core.tensor(np.asarray(config.stdev, dtype=np.float32))
            ),
            (1, -1),
        )
        population = mean + stdev * random.normal(
            k_pop, (pop_size, dim), 0.0, 1.0, "float32"
        )
        population = clamp(population, lb, ub)
    else:
        population = random.uniform(k_pop, (pop_size, dim), 0.0, 1.0, "float32")
        population = population * (ub - lb) + lb

    F_u = enp.full((pop_size,), 0.5, dtype="float32")
    CR_u = enp.full((pop_size,), 0.5, dtype="float32")
    fit = enp.full((pop_size,), float("inf"), dtype="float32")
    F_vec = enp.zeros((pop_size,), dtype="float32")
    CR_vec = enp.zeros((pop_size,), dtype="float32")
    return JaDEState(
        pop=population,
        fit=fit,
        F_u=F_u,
        CR_u=CR_u,
        trial_vectors=population,
        F_vec=F_vec,
        CR_vec=CR_vec,
        key=key,
    )


def ask(config: JaDE, state: JaDEState) -> Tuple[Tensor, JaDEState]:
    """Mutation + crossover: draw F/CR, build trial vectors; keep them for tell."""
    pop, fit, F_u, CR_u = state.pop, state.fit, state.F_u, state.CR_u
    pop_size, dim = config.pop_size, len(config.lb)
    lb, ub = _bake_bounds(config)

    # 1) Generate current F_vec and CR_vec with adaptive perturbation
    key, k_f = random.split(state.key)
    F_vec = clamp_float(
        random.normal(k_f, (pop_size,), 0.0, 1.0, "float32") * 0.1 + F_u, 0.0, 1.0
    )
    key, k_cr = random.split(key)
    CR_vec = clamp_float(
        random.normal(k_cr, (pop_size,), 0.0, 1.0, "float32") * 0.1 + CR_u, 0.0, 1.0
    )

    # 2) Mutation: difference vectors + pbest base vectors
    num_vec = config.num_difference_vectors * 2 + 1
    random_choices = []
    for _ in range(num_vec):
        key, k_choice = random.split(key)
        random_choices.append(random.randint(k_choice, (pop_size,), 0, pop_size, "int32"))

    difference_vectors = etl.sum(
        etl.stack(
            [
                etl.gather(pop, random_choices[i], axis=0)
                - etl.gather(pop, random_choices[i + 1], axis=0)
                for i in range(1, num_vec - 1, 2)
            ],
            axis=0,
        ),
        axes=0,
    )

    key, k_pbest = random.split(key)
    pbest_vectors = select_rand_pbest(k_pbest, 0.05, pop, fit)
    F_vec_2D = enp.expand_dims(F_vec, 1)
    base_vectors = pop + F_vec_2D * (pbest_vectors - pop)
    mutation_vectors = base_vectors + difference_vectors * F_vec_2D

    # 3) Crossover: binomial with guaranteed one mutated dimension per individual
    key, k_cross = random.split(key)
    cross_prob = random.uniform(k_cross, (pop_size, dim), 0.0, 1.0, "float32")
    key, k_dim = random.split(key)
    random_dim = random.randint(k_dim, (pop_size, 1), 0, dim, "int32")
    mask = etl.logical_or(
        cross_prob < enp.expand_dims(CR_vec, 1),
        enp.expand_dims(enp.arange(dim, dtype="int32"), 0) == random_dim,
    )
    new_population = etl.select(mask, mutation_vectors, pop)
    new_population = clamp(new_population, lb, ub)

    new_state = JaDEState(
        pop=pop,
        fit=fit,
        F_u=F_u,
        CR_u=CR_u,
        trial_vectors=new_population,
        F_vec=F_vec,
        CR_vec=CR_vec,
        key=key,
    )
    return new_population, new_state


def tell(config: JaDE, state: JaDEState, fitness: Tensor) -> JaDEState:
    """Selection over trial vectors, then F_u/CR_u adaptation from successes."""
    compare = fitness < state.fit
    pop = etl.select(enp.expand_dims(compare, 1), state.trial_vectors, state.pop)
    fit = etl.select(compare, fitness, state.fit)

    compare_float = etl.cast(compare, "float32")
    F_vec, CR_vec = state.F_vec, state.CR_vec
    sum_F2 = etl.sum(F_vec**2 * compare_float, axes=0)
    sum_F = etl.sum(F_vec * compare_float, axes=0)
    sum_CR = etl.sum(CR_vec * compare_float, axes=0)
    count = etl.sum(compare_float, axes=0)

    mean_F_success = etl.select(count > 0, sum_F2 / (sum_F + 1e-9), 0.0)
    mean_CR_success = etl.select(count > 0, sum_CR / (count + 1e-9), 0.0)

    updated_F_u = (1 - config.c) * state.F_u + config.c * mean_F_success
    updated_CR_u = (1 - config.c) * state.CR_u + config.c * mean_CR_success

    count_mask = count > 0.0
    F_u = etl.select(count_mask, updated_F_u, state.F_u)
    CR_u = etl.select(count_mask, updated_CR_u, state.CR_u)

    return JaDEState(
        pop=pop,
        fit=fit,
        F_u=F_u,
        CR_u=CR_u,
        trial_vectors=state.trial_vectors,
        F_vec=F_vec,
        CR_vec=CR_vec,
        key=state.key,
    )
