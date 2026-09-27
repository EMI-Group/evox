"""Smoke test for the functional NSGA2 port (``evox_etl.algorithms.mo.nsga2``).

ETL-only, no torch. Checks the step-protocol gen-0 contract (``init_step``
evaluates the FULL initial population, later ``step`` calls evaluate the
offspring batch they generate), a 3-generation DTLZ1 run via
``helpers.run_generations``, seed determinism, and the optional custom
operator injection (``NSGA2Config.selection_op``/``crossover_op``/
``mutation_op``; ``None`` means the torch default).
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
from evox_etl.algorithms.mo import nsga2
from helpers import DTLZ1Config, run_generations

POP_SIZE, N_OBJS, DIM, N_GENS, SEED = 20, 3, 7, 3, 0


def make_config(**ops):
    """NSGA2 config: 3 objectives, decision space [0, 1]^7.

    ``ops`` forwards optional custom ``selection_op``/``mutation_op``/
    ``crossover_op`` to ``make_nsga2`` (``None`` = torch default).
    """
    return nsga2.make_nsga2(
        pop_size=POP_SIZE,
        n_objs=N_OBJS,
        lb=np.zeros(DIM, dtype=np.float32),
        ub=np.ones(DIM, dtype=np.float32),
        **ops,
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
    init_exe = etl.build(nsga2.init, cfg, key_spec, backend="numpy")
    state = etl.run(init_exe, cfg, np.asarray(SEED, dtype=np.int64))
    assert tuple(state.pop.shape) == (POP_SIZE, DIM)

    # Gen 0 (init_step) evaluates the FULL initial population (no RNG draw):
    # exactly one evaluate call, on the pop itself, and the resulting fit is
    # the recorded fitness of that batch (finite; rank/dis filled in).
    pop_before = state.pop
    (state, batch), seen = _trace_step(nsga2.init_step, cfg, state)
    assert seen == [(POP_SIZE, DIM)]
    assert tuple(state.pop.shape) == (POP_SIZE, DIM)
    assert np.array_equal(np.asarray(batch.numpy()), np.asarray(pop_before.numpy()))
    assert np.all(np.isfinite(np.asarray(state.fit.numpy())))

    # The first regular step generates its own offspring batch and evaluates
    # THAT batch (not the parents): one evaluate call on (POP_SIZE, DIM)
    # candidates that differ from the parents.
    pop_before = state.pop
    (state, batch), seen = _trace_step(nsga2.step, cfg, state)
    assert seen == [(POP_SIZE, DIM)]
    assert tuple(batch.shape) == (POP_SIZE, DIM)
    assert not np.array_equal(np.asarray(batch.numpy()), np.asarray(pop_before.numpy()))
    # Fused-step contract: the evaluated batch IS the offspring stored in the
    # state by the generation stage of this same call.
    assert np.array_equal(np.asarray(batch.numpy()), np.asarray(state.offspring.numpy()))


def test_full_run_shapes_and_sanity():
    cfg = make_config()
    state = run_generations(nsga2, cfg, DTLZ1Config(DIM, N_OBJS), n_gens=N_GENS, seed=SEED)
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
    s1 = run_generations(nsga2, cfg, DTLZ1Config(DIM, N_OBJS), n_gens=N_GENS, seed=SEED)
    s2 = run_generations(nsga2, cfg, DTLZ1Config(DIM, N_OBJS), n_gens=N_GENS, seed=SEED)
    assert np.array_equal(np.asarray(s1.pop.numpy()), np.asarray(s2.pop.numpy()))


# ---------------------------------------------------------------------------
# Custom operator injection (config op fields, torch signature parity)
# ---------------------------------------------------------------------------


def _init_state(cfg):
    """Run init + init_step on ``cfg`` and return a ready-to-step state."""
    key_spec = etl.core.TensorSpec(shape=(), dtype=np.dtype("int64"))
    init_exe = etl.build(nsga2.init, cfg, key_spec, backend="numpy")
    state = etl.run(init_exe, cfg, np.asarray(SEED, dtype=np.int64))
    (state, _), _ = _trace_step(nsga2.init_step, cfg, state)
    return state


def test_custom_ops_are_invoked_and_delegate_to_defaults():
    calls = {"selection": 0, "crossover": 0, "mutation": 0}

    def selection_op(key, n_parents, fitness):
        calls["selection"] += 1
        return nsga2.tournament_selection_multifit(key, n_parents, fitness)

    def crossover_op(key, x):
        calls["crossover"] += 1
        return nsga2.simulated_binary(key, x)

    def mutation_op(key, x, lb, ub):
        calls["mutation"] += 1
        return nsga2.polynomial_mutation(key, x, lb, ub)

    base_state = _init_state(make_config())
    custom_cfg = make_config(
        selection_op=selection_op, mutation_op=mutation_op, crossover_op=crossover_op
    )
    (custom_state, _), _ = _trace_step(nsga2.step, custom_cfg, base_state)
    (default_state, _), _ = _trace_step(nsga2.step, make_config(), base_state)

    # Each user-supplied op is actually invoked while tracing/running a step.
    assert calls["selection"] >= 1
    assert calls["crossover"] >= 1
    assert calls["mutation"] >= 1
    # Ops that delegate to the etl defaults reproduce the default run exactly.
    assert np.array_equal(
        np.asarray(custom_state.offspring.numpy()),
        np.asarray(default_state.offspring.numpy()),
    )


def test_identity_mutation_changes_offspring_and_none_reproduces_default():
    base_state = _init_state(make_config())
    box = {}

    def identity_mutation(key, x, lb, ub):
        box["mutation_in"] = x
        return x

    prob_cfg = DTLZ1Config(DIM, N_OBJS)

    def generation_fn(state):
        def evaluate(candidates):
            fitness, _ = helpers.toy_evaluate(
                prob_cfg, helpers.ToyProblemState(), candidates
            )
            return fitness

        return (
            nsga2.step(make_config(mutation_op=identity_mutation), state, evaluate),
            box["mutation_in"],
        )

    exe = etl.build(generation_fn, _spec_tree(base_state), backend="numpy")
    ident_state, mutation_in = etl.run(exe, base_state)
    offspring = np.asarray(ident_state.offspring.numpy())

    # Identity mutation => stored offspring equals the (clamped) crossover batch.
    assert np.allclose(offspring, np.clip(np.asarray(mutation_in.numpy()), 0.0, 1.0))

    # Same seed with the DEFAULT mutation gives a different offspring batch.
    (default_state, _), _ = _trace_step(nsga2.step, make_config(), base_state)
    assert not np.allclose(offspring, np.asarray(default_state.offspring.numpy()))

    # An explicit mutation_op=None resolves to the default (no change at all).
    (none_state, _), _ = _trace_step(
        nsga2.step, make_config(mutation_op=None), base_state
    )
    assert np.array_equal(
        np.asarray(none_state.offspring.numpy()),
        np.asarray(default_state.offspring.numpy()),
    )


def test_custom_crossover_op_affects_offspring():
    base_state = _init_state(make_config())

    def constant_crossover(key, x):
        return x * 0.0 + 0.5

    cfg = make_config(
        crossover_op=constant_crossover, mutation_op=lambda key, x, lb, ub: x
    )
    (state, _), _ = _trace_step(nsga2.step, cfg, base_state)
    offspring = np.asarray(state.offspring.numpy())
    # Constant crossover + identity mutation => constant (clamped) offspring.
    assert np.allclose(offspring, 0.5)

    (default_state, _), _ = _trace_step(nsga2.step, make_config(), base_state)
    assert not np.allclose(offspring, np.asarray(default_state.offspring.numpy()))


def test_make_nsga2_rejects_non_callable_ops():
    lb = np.zeros(DIM, dtype=np.float32)
    ub = np.ones(DIM, dtype=np.float32)
    for kw in ("selection_op", "mutation_op", "crossover_op"):
        with pytest.raises(ValueError):
            nsga2.make_nsga2(POP_SIZE, N_OBJS, lb, ub, **{kw: 123})
