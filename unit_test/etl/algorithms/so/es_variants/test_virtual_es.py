"""Unit tests for the functional VirtualES port (ETL numpy backend, no torch).

Coverage:
1. `make_virtual_es` normalization + validation (ValueError, never bare assert).
2. The `init` state contract: leaf shapes/dtypes and the initial center.
3. A single `step` over the module-scope virtual problem: the seeds are
   resampled and the center moves.
4. A self-contained end-to-end convergence smoke through `StdWorkflow`, driven
   by a MINIMAL in-test virtual-population problem defined at MODULE scope (the
   workflow resolves `evaluate` via ``type(config).__module__``, so the module
   must be importable and expose `evaluate`).

Noise budget note: the VirtualES gradient estimate feeds the fitness magnitude
back into its own noise (variance grows like ``||center||**4``), so a tiny
population is numerically UNSTABLE on a plain quadratic objective — the
generation count and `pop_size` below are chosen to stay comfortably inside the
convergent regime (see the module docstring of the LoRA test for the same
effect).
"""

import math
from dataclasses import dataclass

import numpy as np
import pytest

import etl
import etl.numpy as enp

from evox_etl.algorithms.so.es_variants import virtual_es
from evox_etl.algorithms.so.es_variants.virtual_noise import virtual_normal
from evox_etl.core.workflow import StdWorkflow

#: Parameter-block layout used by every test here (dim = 4 * 3 + 4 = 16).
SHAPES = ((4, 3), (4,))
DIM = 16
POP_SIZE = 512
LEARNING_RATE = 0.05
NOISE_STDEV = 1.0


@dataclass(frozen=True)
class _VirtualSphereConfig:
    """Minimal virtual-population problem: ``loss_i = sum((c + sigma * n_i)**2)``.

    Module scope is REQUIRED: `StdWorkflow` resolves the problem's `evaluate`
    from ``type(config).__module__``.  `param_shapes`/`noise_stdev` are only
    used to regenerate the perturbation for the forward pass — the algorithm
    supplies `center`/`seeds`/`sigma` inside its `(center, seeds, sigma)` payload.
    """

    param_shapes: tuple[tuple[int, ...], ...]
    noise_stdev: float


def evaluate(config: _VirtualSphereConfig, state: object, payload: object):
    """Evaluate the ``(center, seeds, sigma)`` payload -> ``((pop,), state)``.

    The perturbation is rebuilt with the SAME `virtual_normal` call the
    algorithm's gradient uses, so the forward pass and the gradient estimate
    are consistent (element offset 0 over the whole flat parameter vector).
    """
    center, seeds, sigma = payload
    dim = sum(math.prod(shape) for shape in config.param_shapes)
    noise = virtual_normal(seeds, 0, dim)  # (pop_size, dim)
    perturbed = enp.expand_dims(center, axis=0) + sigma * noise
    return etl.sum(perturbed * perturbed, axes=1), state


def _config(**overrides: object) -> virtual_es.VirtualESConfig:
    """Build a valid config, overriding individual fields."""
    kwargs: dict = dict(
        param_shapes=SHAPES,
        pop_size=POP_SIZE,
        center_init=np.ones(DIM, np.float32),
        learning_rate=LEARNING_RATE,
        noise_stdev=NOISE_STDEV,
    )
    kwargs.update(overrides)
    return virtual_es.make_virtual_es(**kwargs)


def _init_state(cfg: virtual_es.VirtualESConfig, seed: int = 0):
    """Run the traced `init` graph once and return the resulting state."""
    exe = etl.build(
        virtual_es.init, cfg, etl.core.TensorSpec((), np.int64), backend="numpy"
    )
    return etl.run(exe, cfg, np.asarray(seed, dtype=np.int64))


# --------------------------------------------------------------------- make_*


def test_make_virtual_es_normalizes_config():
    """Array-like inputs are normalized; `dim` is derived eagerly."""
    cfg = virtual_es.make_virtual_es(
        param_shapes=[[4, 3], [4]],
        pop_size=8,
        center_init=[1] * DIM,
        learning_rate=LEARNING_RATE,
        noise_stdev=NOISE_STDEV,
    )
    assert cfg.param_shapes == ((4, 3), (4,))
    assert isinstance(cfg.center_init, tuple) and len(cfg.center_init) == DIM
    assert all(isinstance(v, float) for v in cfg.center_init)
    assert cfg.dim == DIM
    assert cfg.optimizer is None


def test_torch_parity_aliases():
    """`VirtualLoRAES = VirtualES` alias mirrors torch `virtual_es.py:122`."""
    assert virtual_es.VirtualES is virtual_es.VirtualESConfig
    assert virtual_es.VirtualLoRAES is virtual_es.VirtualES


@pytest.mark.parametrize(
    "overrides",
    [
        {"learning_rate": 0.0},
        {"learning_rate": -0.01},
        {"noise_stdev": 0.0},
        {"noise_stdev": -1.0},
        {"pop_size": 0},
        {"pop_size": -4},
        {"optimizer": "sgd"},
    ],
)
def test_make_virtual_es_rejects_invalid_hyperparameters(overrides):
    with pytest.raises(ValueError):
        _config(**overrides)


def test_make_virtual_es_rejects_center_init_length_mismatch():
    with pytest.raises(ValueError, match="center_init length"):
        _config(center_init=np.ones(DIM + 1, np.float32))


@pytest.mark.parametrize(
    "param_shapes",
    [None, (), [], [()], [(4, 0)], [(4, -1)], [(4, 3), (0,)]],
)
def test_make_virtual_es_rejects_invalid_param_shapes(param_shapes):
    with pytest.raises(ValueError):
        _config(param_shapes=param_shapes)


# ----------------------------------------------------------------------- init


def test_init_state_shapes_and_dtypes():
    """`init` returns tensor-only leaves with the documented shapes/dtypes."""
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

    best = state.best_fitness.numpy()
    assert best.shape == ()
    assert best.dtype == np.float32
    assert best == np.inf

    key = state.key.numpy()
    assert key.shape == ()
    assert key.dtype == np.int64


# ----------------------------------------------------------------------- step


def _step_once(cfg: virtual_es.VirtualESConfig, state):
    """Run one traced `step` against the module-scope virtual problem."""

    def generation(alg_state):
        problem = _VirtualSphereConfig(cfg.param_shapes, cfg.noise_stdev)

        def evaluate_closure(payload):
            fitness, _ = evaluate(problem, None, payload)
            return fitness

        return virtual_es.step(cfg, alg_state, evaluate_closure)

    specs = etl.tree_map(
        lambda t: etl.core.TensorSpec(tuple(t.shape), t.dtype), state
    )
    return etl.run(etl.build(generation, specs, backend="numpy"), state)


def test_step_resamples_seeds_and_moves_the_center():
    cfg = _config(pop_size=8)
    state = _init_state(cfg)
    nxt = _step_once(cfg, state)

    assert nxt.center.numpy().shape == (DIM,)
    assert not np.array_equal(nxt.seeds.numpy(), state.seeds.numpy())
    assert not np.allclose(nxt.center.numpy(), state.center.numpy())
    assert np.isfinite(float(nxt.best_fitness.numpy()))
    # optimizer is None -> the Adam moments are passed through untouched
    np.testing.assert_array_equal(nxt.exp_avg.numpy(), np.zeros(DIM, np.float32))


# ---------------------------------------------------------------- convergence


def test_virtual_es_converges_on_perturbed_sphere():
    """End-to-end `StdWorkflow` smoke: the center's squared norm must drop."""
    cfg = _config()
    problem = _VirtualSphereConfig(cfg.param_shapes, cfg.noise_stdev)
    workflow = StdWorkflow(cfg, problem, opt_direction="min")

    state = workflow.run(generations=60, seed=0)
    center = np.asarray(state.algorithm_state.center.numpy(), np.float64)
    initial = float(np.sum(np.ones(DIM, np.float64) ** 2))
    final = float(np.sum(center * center))
    best = float(state.algorithm_state.best_fitness.numpy())

    assert np.isfinite(final), "the ES update diverged — check pop_size/lr"
    assert final < 0.25 * initial, f"expected convergence, got {final} vs {initial}"
    assert 0.0 <= best < initial
    # no optimizer -> both Adam moments are the untouched zeros from `init`
    assert not np.any(state.algorithm_state.exp_avg.numpy())
    assert not np.any(state.algorithm_state.exp_avg_sq.numpy())


def test_virtual_es_adam_converges_and_updates_moments():
    cfg = _config(optimizer="adam")
    problem = _VirtualSphereConfig(cfg.param_shapes, cfg.noise_stdev)
    workflow = StdWorkflow(cfg, problem, opt_direction="min")

    state = workflow.run(generations=40, seed=0)
    center = np.asarray(state.algorithm_state.center.numpy(), np.float64)
    initial = float(DIM)
    final = float(np.sum(center * center))

    assert np.isfinite(final)
    assert final < 0.5 * initial, f"expected convergence, got {final} vs {initial}"
    # the Adam branch must populate BOTH moment vectors
    assert np.any(state.algorithm_state.exp_avg.numpy())
    assert np.any(state.algorithm_state.exp_avg_sq.numpy() > 0)
