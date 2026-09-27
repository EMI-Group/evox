"""Unit tests for the functional VirtualLoRAES port (ETL numpy backend, no torch).

This is a DISTINCT low-rank variant of VirtualES: a >= 2-D parameter block is
perturbed by ``delta = B @ A`` (``A`` ``(rank, k)``, ``B`` ``(d, rank)``) while
1-D blocks keep full Gaussian noise.  Coverage:

1. `make_virtual_lora_es` normalization + validation (including `lora_rank`).
2. The `init` state contract: leaf shapes/dtypes.
3. The two perturbation paths of `lora_factors` (tuple for >= 2-D, flat tensor
   for 1-D) and the batched `dot(B, A)` delta vs a host-side numpy matmul.
4. A self-contained end-to-end convergence smoke through `StdWorkflow` driven
   by a MINIMAL in-test virtual-LoRA problem defined at MODULE scope (the
   workflow resolves `evaluate` via ``type(config).__module__``).

Noise budget note: the low-rank estimator has a LARGER variance than VirtualES
(``delta`` entries have variance ``rank`` and heavier tails), so the population
is larger than the VirtualES test's to stay inside the convergent regime — the
plain quadratic objective feeds its own fitness magnitude back into the
estimator noise.
"""

import math
from dataclasses import dataclass

import numpy as np
import pytest

import etl
import etl.numpy as enp

from evox_etl.algorithms.so.es_variants import virtual_lora_es
from evox_etl.algorithms.so.es_variants.virtual_noise import (
    compute_counter_offsets,
    lora_factors,
)
from evox_etl.core.workflow import StdWorkflow

#: Parameter-block layout used by every test here (dim = 4 * 3 + 4 = 16).
SHAPES = ((4, 3), (4,))
DIM = 16
LORA_RANK = 2
POP_SIZE = 1024
LEARNING_RATE = 0.05
NOISE_STDEV = 1.0


@dataclass(frozen=True)
class _VirtualLoRASphereConfig:
    """Minimal virtual-LoRA problem: ``loss_i = sum((c + sigma * delta_i)**2)``.

    Module scope is REQUIRED (`StdWorkflow` resolves `evaluate` through
    ``type(config).__module__``).  The per-individual perturbation is
    reassembled from the same `lora_factors` calls the algorithm's gradient
    uses, so forward pass and gradient stay consistent.
    """

    param_shapes: tuple[tuple[int, ...], ...]
    noise_stdev: float
    lora_rank: int


def _perturbation(config: _VirtualLoRASphereConfig, seeds, pop_size: int):
    """Flat ``(pop_size, dim)`` per-individual perturbation for `config`."""
    parts = []
    counters = compute_counter_offsets(config.param_shapes, config.lora_rank)
    for shape, counter in zip(config.param_shapes, counters):
        factors = lora_factors(seeds, shape, config.lora_rank, counter)
        if isinstance(factors, tuple):
            a, b = factors  # (pop, rank, k), (pop, d, rank)
            parts.append(enp.reshape(etl.dot(b, a), (pop_size, -1)))
        else:
            parts.append(factors)  # (pop, n) flat noise
    return etl.concatenate(parts, axis=1)


def evaluate(config: _VirtualLoRASphereConfig, state: object, payload: object):
    """Evaluate the ``(center, seeds, sigma)`` payload -> ``((pop,), state)``."""
    center, seeds, sigma = payload
    noise = _perturbation(config, seeds, int(seeds.shape[0]))
    perturbed = enp.expand_dims(center, axis=0) + sigma * noise
    return etl.sum(perturbed * perturbed, axes=1), state


def _config(**overrides: object) -> virtual_lora_es.VirtualLoRAESConfig:
    """Build a valid config, overriding individual fields."""
    kwargs: dict = dict(
        param_shapes=SHAPES,
        lora_rank=LORA_RANK,
        pop_size=POP_SIZE,
        center_init=np.ones(DIM, np.float32),
        learning_rate=LEARNING_RATE,
        noise_stdev=NOISE_STDEV,
    )
    kwargs.update(overrides)
    return virtual_lora_es.make_virtual_lora_es(**kwargs)


def _init_state(cfg: virtual_lora_es.VirtualLoRAESConfig, seed: int = 0):
    """Run the traced `init` graph once and return the resulting state."""
    exe = etl.build(
        virtual_lora_es.init, cfg, etl.core.TensorSpec((), np.int64), backend="numpy"
    )
    return etl.run(exe, cfg, np.asarray(seed, dtype=np.int64))


# --------------------------------------------------------------------- make_*


def test_make_virtual_lora_es_normalizes_config():
    cfg = virtual_lora_es.make_virtual_lora_es(
        param_shapes=[[4, 3], [4]],
        lora_rank=4,
        pop_size=8,
        center_init=[1] * DIM,
        learning_rate=LEARNING_RATE,
        noise_stdev=NOISE_STDEV,
    )
    assert cfg.param_shapes == ((4, 3), (4,))
    assert cfg.lora_rank == 4
    assert isinstance(cfg.center_init, tuple) and len(cfg.center_init) == DIM
    assert cfg.dim == DIM
    assert cfg.optimizer is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"learning_rate": 0.0},
        {"learning_rate": -0.01},
        {"noise_stdev": 0.0},
        {"noise_stdev": -1.0},
        {"pop_size": 0},
        {"pop_size": -4},
        {"lora_rank": 0},
        {"lora_rank": -2},
        {"optimizer": "sgd"},
    ],
)
def test_make_virtual_lora_es_rejects_invalid_hyperparameters(overrides):
    with pytest.raises(ValueError):
        _config(**overrides)


def test_make_virtual_lora_es_rejects_center_init_length_mismatch():
    with pytest.raises(ValueError, match="center_init length"):
        _config(center_init=np.ones(DIM + 3, np.float32))


@pytest.mark.parametrize(
    "param_shapes",
    [None, (), [], [()], [(4, 0)], [(4, -1)], [(4, 3), (0,)]],
)
def test_make_virtual_lora_es_rejects_invalid_param_shapes(param_shapes):
    with pytest.raises(ValueError):
        _config(param_shapes=param_shapes)


# ----------------------------------------------------------------------- init


def test_init_state_shapes_and_dtypes():
    cfg = _config(pop_size=8)
    state = _init_state(cfg)

    center = state.center.numpy()
    assert center.shape == (DIM,)
    assert center.dtype == np.float32
    np.testing.assert_allclose(center, np.ones(DIM, np.float32))

    seeds = state.seeds.numpy()
    assert seeds.shape == (8,)
    assert seeds.dtype == np.int64
    assert np.all(seeds >= 0) and np.all(seeds < 2**31)

    for leaf in (state.exp_avg, state.exp_avg_sq):
        arr = leaf.numpy()
        assert arr.shape == (DIM,)
        assert arr.dtype == np.float32
        np.testing.assert_array_equal(arr, np.zeros(DIM, np.float32))

    assert state.best_fitness.numpy().shape == ()
    assert state.best_fitness.numpy().dtype == np.float32
    assert state.best_fitness.numpy() == np.inf
    assert state.key.numpy().shape == ()
    assert state.key.numpy().dtype == np.int64


# ------------------------------------------------------- low-rank noise paths


def test_lora_factors_exercises_both_block_paths():
    """>= 2-D block -> (A, B) tuple; 1-D block -> flat (pop, n) tensor."""
    pop_size = 8
    counters = compute_counter_offsets(SHAPES, LORA_RANK)

    def factors(seeds):
        a, b = lora_factors(seeds, SHAPES[0], LORA_RANK, counters[0])
        flat = lora_factors(seeds, SHAPES[1], LORA_RANK, counters[1])
        return a, b, flat

    exe = etl.build(
        factors, etl.core.TensorSpec((pop_size,), np.int64), backend="numpy"
    )
    a, b, flat = etl.run(exe, np.arange(pop_size, dtype=np.int64))

    assert a.numpy().shape == (pop_size, LORA_RANK, 3)
    assert b.numpy().shape == (pop_size, 4, LORA_RANK)
    assert flat.numpy().shape == (pop_size, 4)
    assert a.numpy().dtype == np.float32
    assert flat.numpy().dtype == np.float32


def test_lora_delta_is_a_true_batched_matmul():
    """The traced `dot(B, A)` delta equals a host-side batched numpy matmul."""
    pop_size = 8
    counter = compute_counter_offsets(SHAPES, LORA_RANK)[0]

    def delta(seeds):
        a, b = lora_factors(seeds, SHAPES[0], LORA_RANK, counter)
        return a, b, etl.dot(b, a)

    exe = etl.build(
        delta, etl.core.TensorSpec((pop_size,), np.int64), backend="numpy"
    )
    a, b, traced = etl.run(exe, np.arange(pop_size, dtype=np.int64))

    expected = b.numpy() @ a.numpy()  # (pop, d, rank) @ (pop, rank, k)
    assert traced.numpy().shape == (pop_size, 4, 3)
    np.testing.assert_allclose(traced.numpy(), expected, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------- convergence


def test_virtual_lora_es_converges_on_perturbed_sphere():
    """End-to-end `StdWorkflow` smoke: the center's squared norm must drop."""
    cfg = _config()
    problem = _VirtualLoRASphereConfig(
        cfg.param_shapes, cfg.noise_stdev, cfg.lora_rank
    )
    workflow = StdWorkflow(cfg, problem, opt_direction="min")

    state = workflow.run(generations=40, seed=0)
    center = np.asarray(state.algorithm_state.center.numpy(), np.float64)
    initial = float(DIM)
    final = float(np.sum(center * center))
    best = float(state.algorithm_state.best_fitness.numpy())

    assert np.isfinite(final), "the low-rank ES update diverged"
    assert final < 0.25 * initial, f"expected convergence, got {final} vs {initial}"
    assert 0.0 <= best < initial
    # no optimizer -> both Adam moments are the untouched zeros from `init`
    assert not np.any(state.algorithm_state.exp_avg.numpy())
    assert not np.any(state.algorithm_state.exp_avg_sq.numpy())


def test_virtual_lora_es_adam_converges_and_updates_moments():
    cfg = _config(optimizer="adam", learning_rate=0.01)
    problem = _VirtualLoRASphereConfig(
        cfg.param_shapes, cfg.noise_stdev, cfg.lora_rank
    )
    workflow = StdWorkflow(cfg, problem, opt_direction="min")

    state = workflow.run(generations=40, seed=0)
    center = np.asarray(state.algorithm_state.center.numpy(), np.float64)
    final = float(np.sum(center * center))

    assert np.isfinite(final)
    assert final < 0.5 * float(DIM), f"expected convergence, got {final}"
    assert np.any(state.algorithm_state.exp_avg.numpy())
    assert np.any(state.algorithm_state.exp_avg_sq.numpy() > 0)


def test_virtual_lora_es_full_rank_block_stays_well_posed():
    """A 2-D block whose `lora_rank` equals its last dim is still valid."""
    shapes = ((3, 3),)
    cfg = virtual_lora_es.make_virtual_lora_es(
        shapes,
        lora_rank=3,
        pop_size=512,
        center_init=np.ones(9, np.float32),
        learning_rate=0.05,
        noise_stdev=1.0,
    )
    problem = _VirtualLoRASphereConfig(shapes, 1.0, 3)
    workflow = StdWorkflow(cfg, problem, opt_direction="min")

    state = workflow.run(generations=40, seed=0)
    center = np.asarray(state.algorithm_state.center.numpy(), np.float64)
    final = float(np.sum(center * center))
    assert np.isfinite(final)
    assert final < 0.5 * 9.0
