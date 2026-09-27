"""Smoke test for the functional NSGA3 port (``evox_etl.algorithms.mo.nsga3``).

ETL-only, no torch. Checks the step-protocol gen-0 contract (``init_step``
evaluates the FULL initial population, later ``step`` calls evaluate the
offspring batch they generate), a 3-generation DTLZ1 run via
``helpers.run_generations``, seed determinism, the ``data_type`` contract
(only ``None``/builtin ``bool`` accepted), and the survivor-ordering parity
of the final selection stage.
"""

import pathlib
import sys

_PATH = pathlib.Path(__file__).resolve()
# NOTE: this file sits one level deeper than the other algorithms tests
# (unit_test/etl/algorithms/mo/), so the repository root is parents[4].
_ROOT = _PATH.parents[4]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_PATH.parents[1]))  # unit_test/etl/algorithms -> helpers

import numpy as np
import pytest

import etl
import etl.numpy as enp

import helpers
from evox_etl.algorithms.mo import nsga3
from helpers import DTLZ1Config, run_generations

POP_SIZE, N_OBJS, DIM, N_GENS, SEED = 20, 3, 7, 3, 0


def make_config():
    """NSGA3 config: 3 objectives, decision space [0, 1]^7."""
    return nsga3.make_nsga3(
        pop_size=POP_SIZE,
        n_objs=N_OBJS,
        lb=np.zeros(DIM, dtype=np.float32),
        ub=np.ones(DIM, dtype=np.float32),
    )


def _leaf_spec(t):
    """TensorSpec mirroring one etl tensor leaf."""
    return etl.core.TensorSpec(shape=tuple(t.shape), dtype=np.dtype(t.dtype))


def _spec_tree(pytree):
    """Pytree of etl tensors -> pytree of TensorSpecs (state mirror)."""
    return etl.tree_map(_leaf_spec, pytree)


def _trace_step(step_fn, cfg, state):
    """Trace one step-family call ``step_fn(cfg, state, evaluate)`` with an
    ``evaluate`` closure that evaluates DTLZ1 AND records the candidate batch
    it received, returning ``(state, batch)`` — the batch is a graph output
    computed inside the same trace, so the record is exact (no host-side
    approximation)."""
    prob_cfg = DTLZ1Config(DIM, N_OBJS)
    seen = []
    box = {}

    def generation_fn(state):
        def evaluate(candidates):
            seen.append(tuple(candidates.shape))
            box["batch"] = candidates
            fitness, _ = helpers.toy_evaluate(prob_cfg, helpers.ToyProblemState(), candidates)
            return fitness

        return step_fn(cfg, state, evaluate), box["batch"]

    exe = etl.build(generation_fn, _spec_tree(state), backend="numpy")
    return etl.run(exe, state), seen


def test_init_step_full_population_and_step_offspring_shape():
    cfg = make_config()
    key_spec = etl.core.TensorSpec(shape=(), dtype=np.dtype("int64"))
    init_exe = etl.build(nsga3.init, cfg, key_spec, backend="numpy")
    state = etl.run(init_exe, cfg, np.asarray(SEED, dtype=np.int64))
    assert tuple(state.pop.shape) == (POP_SIZE, DIM)

    # Gen 0 (init_step) evaluates the FULL initial population (no RNG draw):
    # exactly one evaluate call, on the pop itself.
    pop_before = state.pop
    (state, batch), seen = _trace_step(nsga3.init_step, cfg, state)
    assert seen == [(POP_SIZE, DIM)]
    assert tuple(state.pop.shape) == (POP_SIZE, DIM)
    assert np.array_equal(np.asarray(batch.numpy()), np.asarray(pop_before.numpy()))
    assert np.all(np.isfinite(np.asarray(state.fit.numpy())))

    # The first regular step generates its own offspring batch and evaluates
    # THAT batch (not the parents): one evaluate call on (POP_SIZE, DIM)
    # candidates that differ from the parents.
    pop_before = state.pop
    (state, batch), seen = _trace_step(nsga3.step, cfg, state)
    assert seen == [(POP_SIZE, DIM)]
    assert tuple(batch.shape) == (POP_SIZE, DIM)
    assert not np.array_equal(np.asarray(batch.numpy()), np.asarray(pop_before.numpy()))
    # Fused-step contract: the evaluated batch IS the offspring stored in the
    # state by the generation stage of this same call.
    assert np.array_equal(np.asarray(batch.numpy()), np.asarray(state.off.numpy()))


def test_full_run_shapes_and_sanity():
    cfg = make_config()
    state = run_generations(nsga3, cfg, DTLZ1Config(DIM, N_OBJS), n_gens=N_GENS, seed=SEED)
    pop = np.asarray(state.pop.numpy())
    fit = np.asarray(state.fit.numpy())
    assert pop.shape == (POP_SIZE, DIM)
    assert fit.shape == (POP_SIZE, N_OBJS)
    assert np.all(np.isfinite(fit))
    # DTLZ1 optimum is 0, values span ~0-600; after 3 gens each objective's
    # best stays well below 100 (loose sanity bound).
    assert np.all(np.min(fit, axis=0) < 100.0)
    assert np.all((pop >= -1e-6) & (pop <= 1.0 + 1e-6))


def test_full_run_deterministic():
    cfg = make_config()
    s1 = run_generations(nsga3, cfg, DTLZ1Config(DIM, N_OBJS), n_gens=N_GENS, seed=SEED)
    s2 = run_generations(nsga3, cfg, DTLZ1Config(DIM, N_OBJS), n_gens=N_GENS, seed=SEED)
    assert np.array_equal(np.asarray(s1.pop.numpy()), np.asarray(s2.pop.numpy()))


# ---------------------------------------------------------------------------
# data_type contract + survivor-ordering parity (torch divergence fixes)
# ---------------------------------------------------------------------------


def _run_init(cfg):
    """Trace ``nsga3.init`` for ``cfg`` and run it with a fixed seed."""
    key_spec = etl.core.TensorSpec(shape=(), dtype=np.dtype("int64"))
    init_exe = etl.build(nsga3.init, cfg, key_spec, backend="numpy")
    return etl.run(init_exe, cfg, np.asarray(SEED, dtype=np.int64))


def test_data_type_bool_draws_boolean_population():
    """``data_type=bool`` takes the torch ``data_type == torch.bool`` branch
    (uniform > 0.5 boolean population); ``None`` (the default) stays float32."""
    cfg_bool = nsga3.make_nsga3(
        pop_size=POP_SIZE, n_objs=N_OBJS,
        lb=np.zeros(DIM, dtype=np.float32), ub=np.ones(DIM, dtype=np.float32),
        data_type=bool,
    )
    state = _run_init(cfg_bool)
    assert state.pop.shape == (POP_SIZE, DIM)
    assert np.dtype(state.pop.dtype) == np.dtype(bool)
    assert set(np.unique(np.asarray(state.pop.numpy())).tolist()) <= {False, True}

    state_float = _run_init(make_config())  # data_type=None
    assert np.dtype(state_float.pop.dtype) == np.dtype("float32")


@pytest.mark.parametrize("bad", ["bool", "float32", int, np.bool_, True])
def test_data_type_rejects_unsupported_values(bad):
    """Only ``None`` and the builtin ``bool`` type are accepted. In particular
    ``torch.bool`` (a dtype object, hence rejected) must not silently degrade."""
    with pytest.raises(ValueError, match="data_type"):
        nsga3.make_nsga3(
            pop_size=POP_SIZE, n_objs=N_OBJS,
            lb=np.zeros(DIM, dtype=np.float32), ub=np.ones(DIM, dtype=np.float32),
            data_type=bad,
        )


def test_direct_config_construction_validates_data_type():
    """Direct ``NSGA3Config`` construction bypasses ``make_nsga3``; ``init``
    must still fail loudly instead of silently degrading to the float path."""
    cfg = nsga3.NSGA3Config(
        pop_size=POP_SIZE, n_objs=N_OBJS, lb=(0.0,) * DIM, ub=(1.0,) * DIM,
        data_type="bool",
    )
    with pytest.raises(ValueError, match="data_type"):
        nsga3.init(cfg, None)


def test_final_survivors_preserve_merge_order():
    """The final selection stage orders survivors by POSITION in the merged
    arrays — torch's ``merge_pop[rank < worst_rank]`` mask order — NOT by
    ascending rank (the old port sorted by rank)."""
    n, k, m = 5, 3, 2
    merge_pop = np.arange(n * k, dtype=np.float32).reshape(n, k)
    merge_fit = (np.arange(n * m, dtype=np.float32).reshape(n, m) + 1.0) * 10.0
    # worst_rank = 2 -> survivors are positions 0, 1, 3 (in merge order).
    rank = np.asarray([1, 0, 2, 0, 5], dtype=np.int32)

    specs = (
        etl.core.TensorSpec(shape=(n, k), dtype=np.dtype("float32")),
        etl.core.TensorSpec(shape=(n, m), dtype=np.dtype("float32")),
        etl.core.TensorSpec(shape=(n,), dtype=np.dtype("int32")),
    )
    exe = etl.build(
        lambda mp, mf, rk: nsga3._final_survivors(mp, mf, rk, 2, 3),
        *specs, backend="numpy",
    )
    pop, fit, out_rank = etl.run(exe, merge_pop, merge_fit, rank)

    survivors = np.asarray([0, 1, 3], dtype=np.int64)
    # [1, 0, 0] is the merge-order ranking; the old sort would give [0, 0, 1].
    assert np.array_equal(out_rank.numpy(), np.asarray([1, 0, 0], dtype=np.int32))
    assert np.array_equal(pop.numpy(), merge_pop[survivors])
    assert np.array_equal(fit.numpy(), merge_fit[survivors])
