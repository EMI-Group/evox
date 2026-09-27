"""Pure-etl tests for the virtual Gaussian-noise neuroevolution problem.

No torch imports: the expected fitness is computed by an INDEPENDENT pure-numpy
reference that re-derives the same splitmix64 + Box-Muller noise stream and
re-implements the layer-by-layer forward pass / loss.  Everything runs through
``etl.build`` + ``etl.run`` on the default numpy backend.
"""

import dataclasses
import math

import numpy as np
import pytest

import etl
import etl.core

from evox_etl.problems.numerical.state import ProblemState
from evox_etl.problems.neuroevolution.virtual_problem import (
    SUPPORTED_ACTIVATIONS,
    VirtualLoRAProblem,
    VirtualProblem,
    VirtualProblemConfig,
    evaluate,
    make_virtual_problem,
)

# --------------------------------------------------------------------------- #
# independent numpy reference
# --------------------------------------------------------------------------- #

_GAMMA = 0x9E3779B97F4A7C15 - (1 << 64)


def _logical_rshift(x: np.ndarray, shift: int) -> np.ndarray:
    return np.bitwise_and(np.right_shift(x, shift), (1 << (64 - shift)) - 1)


def _splitmix64(z: np.ndarray) -> np.ndarray:
    z = np.bitwise_xor(z, _logical_rshift(z, 30))
    z = z * _GAMMA
    z = np.bitwise_xor(z, _logical_rshift(z, 27))
    z = z * _GAMMA
    z = np.bitwise_xor(z, _logical_rshift(z, 31))
    return z


def virtual_normal_np(seeds: np.ndarray, offset: int, n_elements: int) -> np.ndarray:
    """Re-derive the shared splitmix64 + Box-Muller noise stream, float32."""
    seeds = np.asarray(seeds, dtype=np.int64).reshape(-1)
    flat = np.arange(n_elements, dtype=np.int64) + np.int64(offset)
    with np.errstate(over="ignore"):
        z = np.bitwise_xor(seeds[:, None], flat[None, :] * _GAMMA)
        z = _splitmix64(z)
    high32 = np.bitwise_and(np.right_shift(z, 32), 0xFFFFFFFF)
    low32 = np.bitwise_and(z, 0xFFFFFFFF)
    u1 = low32.astype(np.float32) * (1.0 / float(1 << 32))
    u2 = high32.astype(np.float32) * (1.0 / float(1 << 32))
    r = np.sqrt(-2.0 * np.log(np.clip(u1, 1e-10, 1.0)))
    return (r * np.cos(2.0 * np.pi * u2)).astype(np.float32)


def _param_offsets(shapes) -> list[int]:
    offsets, cur = [], 0
    for shape in shapes:
        offsets.append(cur)
        cur += int(np.prod(shape))
    return offsets


def _counter_offsets(shapes, rank: int) -> list[int]:
    offsets, cur = [], 0
    for shape in shapes:
        offsets.append(cur)
        if len(shape) == 1:
            n = int(shape[0])
        else:
            d = int(np.prod(shape[:-1]))
            n = rank * int(shape[-1]) + d * rank
        cur += ((n + 3) // 4) * 4
    return offsets


def _act(name: str, x: np.ndarray) -> np.ndarray:
    if name == "identity":
        return x
    if name == "relu":
        return np.maximum(x, 0.0)
    if name == "tanh":
        return np.tanh(x)
    if name == "sigmoid":
        return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)
    if name == "gelu":
        erf = np.vectorize(math.erf, otypes=[np.float32])
        return (0.5 * x * (1.0 + erf(x / np.float32(math.sqrt(2.0))))).astype(np.float32)
    raise AssertionError(f"unsupported activation {name!r}")


def numpy_fitness(config: VirtualProblemConfig, center: np.ndarray, seeds: np.ndarray, sigma: float):
    """Independent numpy reference for ``evaluate`` (same noise, own forward)."""
    shapes = config.param_shapes
    pop = len(seeds)
    batch = config.batch_size
    p_off = _param_offsets(shapes)
    rank = config.lora_rank
    n_off = p_off if rank is None else _counter_offsets(shapes, rank)

    center = np.asarray(center, dtype=np.float32)
    h = np.asarray(config.inputs, dtype=np.float32).reshape(batch, config.in_features)

    for _, w_idx, b_idx, act in config.layer_specs:
        out_f, in_f = shapes[w_idx]
        w = p_off[w_idx]
        weight = center[w : w + out_f * in_f].reshape(out_f, in_f)
        base = h @ weight.T  # (batch, out) — or (pop, batch, out) after layer 0

        if rank is None:
            noise = virtual_normal_np(seeds, n_off[w_idx], out_f * in_f).reshape(pop, out_f, in_f)
            delta = np.matmul(h, np.transpose(noise, (0, 2, 1)))
        else:
            co = n_off[w_idx]
            a = virtual_normal_np(seeds, co, rank * in_f).reshape(pop, rank, in_f)
            b = virtual_normal_np(seeds, co + ((rank * in_f + 3) // 4) * 4, out_f * rank).reshape(
                pop, out_f, rank
            )
            delta = np.matmul(np.matmul(h, np.transpose(a, (0, 2, 1))), np.transpose(b, (0, 2, 1)))
        out = base + sigma * delta

        if b_idx is not None:
            bo = p_off[b_idx]
            bias = center[bo : bo + out_f]
            bias_noise = virtual_normal_np(seeds, n_off[b_idx], out_f)
            out = out + (bias[None, :] + sigma * bias_noise)[:, None, :]

        h = _act(act, out)

    if config.loss == "mse":
        target = np.asarray(config.targets, dtype=np.float32).reshape(batch, config.out_features)
        diff = h - target
        per_sample = np.mean(diff * diff, axis=2)
    else:
        target = np.asarray(config.targets, dtype=np.int64)
        logit_max = np.max(h, axis=2, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(h - logit_max), axis=2)) + logit_max[:, :, 0]
        selected = h[np.arange(pop)[:, None], np.arange(batch)[None, :], target[None, :]]
        per_sample = log_sum_exp - selected

    if config.reduction == "mean":
        return per_sample.mean(axis=1)
    return per_sample.sum(axis=1)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

POP = 6
SIGMA = 0.1
CENTER_DIM = 49


def base_kwargs():
    """A valid 4 -> 8 -> 1 MLP configuration (torch named_parameters order)."""
    rng = np.random.default_rng(0)
    return {
        "param_shapes": ((8, 4), (8,), (1, 8), (1,)),
        "layer_specs": (("linear", 0, 1, "relu"), ("linear", 2, 3, "identity")),
        "inputs": rng.standard_normal(3 * 4).astype(np.float32),
        "targets": rng.standard_normal(3).astype(np.float32),
        "in_features": 4,
        "out_features": 1,
        "batch_size": 3,
    }


def run_evaluate(config, center, seeds, sigma=SIGMA):
    """Trace ``evaluate`` and run it on the payload; returns cpu fitness + state."""
    payload_spec = (
        etl.core.TensorSpec(center.shape, np.dtype("float32")),
        etl.core.TensorSpec(seeds.shape, np.dtype("int64")),
        sigma,
    )
    exe = etl.build(evaluate, config, ProblemState(), payload_spec)
    fitness, problem_state = etl.run(exe, config, ProblemState(), (center, seeds, sigma))
    return fitness.numpy(), problem_state


def _center_and_seeds(seed=1):
    rng = np.random.default_rng(seed)
    center = rng.standard_normal(CENTER_DIM).astype(np.float32)
    seeds = (np.arange(POP) * 7919 + 13).astype(np.int64)
    return center, seeds


# --------------------------------------------------------------------------- #
# make_virtual_problem validation
# --------------------------------------------------------------------------- #


def test_make_returns_frozen_config():
    cfg = make_virtual_problem(**base_kwargs())
    assert isinstance(cfg, VirtualProblemConfig)
    assert cfg.lora_rank is None
    assert dataclasses.is_dataclass(cfg)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.batch_size = 4  # type: ignore[misc]


def test_class_free_aliases():
    assert VirtualProblem is VirtualProblemConfig
    assert VirtualLoRAProblem is VirtualProblem
    cfg = make_virtual_problem(**base_kwargs())
    # a direct construction from already-normalized plain statics is equivalent
    direct = VirtualProblemConfig(
        param_shapes=cfg.param_shapes,
        layer_specs=cfg.layer_specs,
        inputs=cfg.inputs,
        targets=cfg.targets,
        in_features=cfg.in_features,
        out_features=cfg.out_features,
        batch_size=cfg.batch_size,
    )
    assert direct == cfg

@pytest.mark.parametrize("lora_rank", [2, 4])
def test_make_lora_rank_positive_ok(lora_rank):
    cfg = make_virtual_problem(**base_kwargs(), lora_rank=lora_rank)
    assert cfg.lora_rank == lora_rank


@pytest.mark.parametrize(
    "override",
    [
        {"param_shapes": ((8, 4, 1), (8,), (1, 8), (1,))},  # weight not 2-D
        {"param_shapes": ((8, 4), (9,), (1, 8), (1,))},  # bias shape != (out,)
        {"param_shapes": ()},  # empty shapes
        {"param_shapes": ((8, 4), (8,), (1, 8), (0,))},  # non-positive dim
        {"layer_specs": (("linear", 7, 1, "relu"), ("linear", 2, 3, "identity"))},  # w idx range
        {"layer_specs": (("linear", 0, 7, "relu"), ("linear", 2, 3, "identity"))},  # b idx range
        {"layer_specs": (("dense", 0, 1, "relu"), ("linear", 2, 3, "identity"))},  # bad kind
        {"layer_specs": (("linear", 0, 1, "swish"), ("linear", 2, 3, "identity"))},  # bad act
        {"layer_specs": (("linear", 0, 1), ("linear", 2, 3, "identity"))},  # not a 4-tuple
        {"layer_specs": ()},  # empty
        {"in_features": 5},  # chain mismatch at layer 0
        {"in_features": 0},  # non-positive
        {"out_features": 2},  # last layer mismatch
        {"batch_size": 0},  # non-positive
        {"batch_size": 5},  # inputs length mismatch
        {"inputs": np.zeros(11, dtype=np.float32)},  # inputs length mismatch
        {"targets": np.zeros(4, dtype=np.float32)},  # mse targets length mismatch
        {"reduction": "median"},  # bad reduction
        {"loss": "hinge"},  # bad loss
        {"lora_rank": 0},  # non-positive rank
        {"lora_rank": -3},  # non-positive rank
        {"lora_rank": 1.5},  # non-int rank
    ],
)
def test_make_validation_raises_value_error(override):
    kwargs = base_kwargs()
    kwargs.update(override)
    with pytest.raises(ValueError):
        make_virtual_problem(**kwargs)


def test_make_cross_entropy_targets_length():
    rng = np.random.default_rng(2)
    kwargs = {
        "param_shapes": ((4, 3), (4,), (2, 4), (2,)),
        "layer_specs": (("linear", 0, 1, "relu"), ("linear", 2, 3, "identity")),
        "inputs": rng.standard_normal(3 * 3).astype(np.float32),
        "targets": np.array([0, 1, 0], dtype=np.float32),  # batch_size class indices
        "in_features": 3,
        "out_features": 2,
        "batch_size": 3,
        "loss": "cross_entropy",
    }
    cfg = make_virtual_problem(**kwargs)
    assert cfg.loss == "cross_entropy"
    # the mse-shaped targets (batch_size * out_features) are wrong for cross-entropy
    bad = dict(kwargs)
    bad["targets"] = rng.standard_normal(3 * 2).astype(np.float32)
    with pytest.raises(ValueError):
        make_virtual_problem(**bad)

def test_make_normalizes_to_float32_tuples():
    cfg = make_virtual_problem(**base_kwargs())
    assert isinstance(cfg.inputs, tuple) and isinstance(cfg.targets, tuple)
    assert all(isinstance(v, float) for v in cfg.inputs)
    expected = np.asarray(base_kwargs()["inputs"], dtype=np.float32).tolist()
    assert cfg.inputs == tuple(expected)


# --------------------------------------------------------------------------- #
# correctness vs the independent numpy reference
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("activation", ["relu", "tanh", "sigmoid", "gelu", "identity"])
def test_full_noise_matches_reference(reduction, activation):
    kwargs = base_kwargs()
    kwargs["layer_specs"] = (("linear", 0, 1, activation), ("linear", 2, 3, "identity"))
    kwargs["reduction"] = reduction
    cfg = make_virtual_problem(**kwargs)
    center, seeds = _center_and_seeds()

    fitness, problem_state = run_evaluate(cfg, center, seeds)
    expected = numpy_fitness(cfg, center, seeds, SIGMA)

    assert fitness.shape == (POP,)
    assert fitness.dtype == np.float32
    assert problem_state == ProblemState()
    np.testing.assert_allclose(fitness, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("rank", [1, 2, 3])
def test_lora_matches_reference(reduction, rank):
    kwargs = base_kwargs()
    kwargs["reduction"] = reduction
    kwargs["lora_rank"] = rank
    kwargs["layer_specs"] = (("linear", 0, 1, "relu"), ("linear", 2, 3, "identity"))
    cfg = make_virtual_problem(**kwargs)
    center, seeds = _center_and_seeds(seed=7)

    fitness, _ = run_evaluate(cfg, center, seeds)
    expected = numpy_fitness(cfg, center, seeds, SIGMA)
    np.testing.assert_allclose(fitness, expected, rtol=1e-5, atol=1e-5)

    # LoRA perturbation must differ from the full-noise perturbation.
    full = make_virtual_problem(**base_kwargs())
    full_fitness, _ = run_evaluate(full, center, seeds)
    assert not np.allclose(fitness, full_fitness)


def test_three_layer_network_matches_reference():
    rng = np.random.default_rng(3)
    kwargs = {
        "param_shapes": ((6, 4), (6,), (5, 6), (5,), (2, 5), (2,)),
        "layer_specs": (
            ("linear", 0, 1, "tanh"),
            ("linear", 2, 3, "relu"),
            ("linear", 4, 5, "identity"),
        ),
        "inputs": rng.standard_normal(3 * 4).astype(np.float32),
        "targets": rng.standard_normal(3 * 2).astype(np.float32),
        "in_features": 4,
        "out_features": 2,
        "batch_size": 3,
    }
    cfg = make_virtual_problem(**kwargs)
    center = rng.standard_normal(6 * 4 + 6 + 5 * 6 + 5 + 2 * 5 + 2).astype(np.float32)
    seeds = np.array([11, 22, 33, 44], dtype=np.int64)

    fitness, _ = run_evaluate(cfg, center, seeds)
    expected = numpy_fitness(cfg, center, seeds, SIGMA)
    np.testing.assert_allclose(fitness, expected, rtol=1e-5, atol=1e-5)


def test_no_bias_layer_matches_reference():
    rng = np.random.default_rng(5)
    kwargs = {
        "param_shapes": ((8, 4), (1, 8), (1,)),
        "layer_specs": (("linear", 0, None, "relu"), ("linear", 1, 2, "identity")),
        "inputs": rng.standard_normal(2 * 4).astype(np.float32),
        "targets": rng.standard_normal(2).astype(np.float32),
        "in_features": 4,
        "out_features": 1,
        "batch_size": 2,
    }
    cfg = make_virtual_problem(**kwargs)
    center = rng.standard_normal(8 * 4 + 1 * 8 + 1).astype(np.float32)
    seeds = np.array([1, 2, 3], dtype=np.int64)

    fitness, _ = run_evaluate(cfg, center, seeds, sigma=0.25)
    expected = numpy_fitness(cfg, center, seeds, 0.25)
    np.testing.assert_allclose(fitness, expected, rtol=1e-5, atol=1e-5)


def test_cross_entropy_matches_reference():
    rng = np.random.default_rng(9)
    kwargs = {
        "param_shapes": ((5, 3), (5,), (4, 5), (4,)),
        "layer_specs": (("linear", 0, 1, "tanh"), ("linear", 2, 3, "identity")),
        "inputs": rng.standard_normal(4 * 3).astype(np.float32),
        "targets": np.array([0, 3, 2, 1], dtype=np.float32),
        "in_features": 3,
        "out_features": 4,
        "batch_size": 4,
        "loss": "cross_entropy",
    }
    cfg = make_virtual_problem(**kwargs)
    center = rng.standard_normal(5 * 3 + 5 + 4 * 5 + 4).astype(np.float32)
    seeds = np.array([5, 6, 7, 8, 9], dtype=np.int64)

    fitness, _ = run_evaluate(cfg, center, seeds)
    expected = numpy_fitness(cfg, center, seeds, SIGMA)
    np.testing.assert_allclose(fitness, expected, rtol=1e-5, atol=1e-5)


def test_center_not_mutated_and_state_passthrough():
    cfg = make_virtual_problem(**base_kwargs())
    center, seeds = _center_and_seeds()
    center_copy = center.copy()
    seeds_copy = seeds.copy()
    _, problem_state = run_evaluate(cfg, center, seeds)
    np.testing.assert_array_equal(center, center_copy)
    np.testing.assert_array_equal(seeds, seeds_copy)
    assert problem_state == ProblemState()


def test_determinism_two_runs_identical():
    cfg = make_virtual_problem(**base_kwargs(), lora_rank=2)
    center, seeds = _center_and_seeds()
    first, _ = run_evaluate(cfg, center, seeds)
    second, _ = run_evaluate(cfg, center, seeds)
    np.testing.assert_array_equal(first, second)


def test_zero_sigma_reduces_to_shared_center():
    """With sigma == 0 every individual collapses onto the un-perturbed network."""
    cfg = make_virtual_problem(**base_kwargs())
    center, seeds = _center_and_seeds()
    fitness, _ = run_evaluate(cfg, center, seeds, sigma=0.0)
    assert np.allclose(fitness, fitness[0])
    expected = numpy_fitness(cfg, center, seeds, 0.0)
    np.testing.assert_allclose(fitness, expected, rtol=1e-5, atol=1e-5)


def test_supported_activations_are_forwardable():
    assert SUPPORTED_ACTIVATIONS == ("relu", "tanh", "sigmoid", "gelu", "identity")
