"""Smoke test for the functional MOEAD port (``evox_etl.algorithms.mo.moead``).

ETL-only, no torch. Checks the step-protocol gen-0 contract (``init_step``
evaluates the FULL initial population, later ``step`` calls evaluate the
offspring batch they generate), a 3-generation DTLZ1 run via
``helpers.run_generations``, and seed determinism.
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

import etl
import etl.numpy as enp

import helpers
from evox_etl.algorithms.mo import moead
from helpers import DTLZ1Config, run_generations

POP_SIZE, N_OBJS, DIM, N_GENS, SEED = 20, 3, 7, 3, 0
# Effective population: uniform_sampling overwrites pop_size=20 with the
# Das-Dennis count for 3 objectives (15) — same as the torch reference.
N_EFF = 15


def make_config():
    """MOEAD config: 3 objectives, decision space [0, 1]^7."""
    return moead.make_moead(
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
    init_exe = etl.build(moead.init, cfg, key_spec, backend="numpy")
    state = etl.run(init_exe, cfg, np.asarray(SEED, dtype=np.int64))
    assert tuple(state.pop.shape) == (N_EFF, DIM)

    # Gen 0 (init_step) evaluates the FULL initial population (one solution
    # per weight vector, N_EFF rows — the Das-Dennis count), not an offspring
    # batch: one evaluate call on the pop itself, whose per-objective min
    # becomes the ideal point z.
    pop_before = state.pop
    (state, batch), seen = _trace_step(moead.init_step, cfg, state)
    assert seen == [(N_EFF, DIM)]
    assert np.array_equal(np.asarray(batch.numpy()), np.asarray(pop_before.numpy()))
    assert np.array_equal(
        np.asarray(state.z.numpy()),
        np.min(np.asarray(state.fit.numpy()), axis=0),
    )

    # The first regular step generates one offspring per weight vector and
    # evaluates THAT batch (not the parents): a single (N_EFF, DIM) evaluate
    # call; the sequential per-i update then runs inside the same trace.
    pop_before = state.pop
    (state, batch), seen = _trace_step(moead.step, cfg, state)
    assert seen == [(N_EFF, DIM)]
    assert tuple(batch.shape) == (N_EFF, DIM)
    assert not np.array_equal(np.asarray(batch.numpy()), np.asarray(pop_before.numpy()))
    # Fused-step contract: the evaluated batch IS the offspring stored in the
    # state by the generation stage of this same call.
    assert np.array_equal(
        np.asarray(batch.numpy()), np.asarray(state.next_generation.numpy())
    )


def test_full_run_shapes_and_sanity():
    cfg = make_config()
    state = run_generations(moead, cfg, DTLZ1Config(DIM, N_OBJS), n_gens=N_GENS, seed=SEED)
    pop = np.asarray(state.pop.numpy())
    fit = np.asarray(state.fit.numpy())
    assert pop.shape == (N_EFF, DIM)
    assert fit.shape == (N_EFF, N_OBJS)
    assert np.all(np.isfinite(fit))
    # DTLZ1 optimum is 0, values span ~0-600; at seed 0 the per-objective
    # minima after 3 gens are ~[2.7, 29.0, 40.5], so bound at 100 (loose).
    assert np.all(np.min(fit, axis=0) < 100.0)
    assert np.all((pop >= -1e-6) & (pop <= 1.0 + 1e-6))


def test_full_run_deterministic():
    cfg = make_config()
    s1 = run_generations(moead, cfg, DTLZ1Config(DIM, N_OBJS), n_gens=N_GENS, seed=SEED)
    s2 = run_generations(moead, cfg, DTLZ1Config(DIM, N_OBJS), n_gens=N_GENS, seed=SEED)
    assert np.array_equal(np.asarray(s1.pop.numpy()), np.asarray(s2.pop.numpy()))
