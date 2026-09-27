"""Smoke test for the functional HypE port (``evox_etl.algorithms.mo.hype``).

ETL-only, no torch. Checks the step-protocol gen-0 contract (``init_step``
evaluates the FULL initial population, later ``step`` calls evaluate the
offspring batch they generate), a 3-generation DTLZ1 run via
``helpers.run_generations``, seed determinism, and the optional custom
crossover/mutation op injection (``HypEConfig.mutation_op``/``crossover_op``;
``None`` means the torch default; ``selection_op`` is intentionally not
exposed because it is inert in the torch reference).
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
from evox_etl.algorithms.mo import hype
from evox_etl.operators.crossover import simulated_binary
from evox_etl.operators.mutation import polynomial_mutation
from helpers import DTLZ1Config, run_generations

POP_SIZE, N_OBJS, DIM, N_GENS, SEED = 20, 3, 7, 3, 0


def make_config(**kwargs):
    """HypE config: 3 objectives, decision space [0, 1]^7.

    Extra keyword args are forwarded to ``make_hype`` — used to inject custom
    ``mutation_op`` / ``crossover_op``.
    """
    return hype.make_hype(
        pop_size=POP_SIZE,
        n_objs=N_OBJS,
        lb=np.zeros(DIM, dtype=np.float32),
        ub=np.ones(DIM, dtype=np.float32),
        **kwargs,
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
    init_exe = etl.build(hype.init, cfg, key_spec, backend="numpy")
    state = etl.run(init_exe, cfg, np.asarray(SEED, dtype=np.int64))
    assert tuple(state.pop.shape) == (POP_SIZE, DIM)

    # Gen 0 (init_step) evaluates the FULL initial population (no RNG draw):
    # exactly one evaluate call, on the pop itself, and ref is derived from
    # that fitness as 1.2 * max over ALL its entries (torch init_step 1:1).
    pop_before = state.pop
    (state, batch), seen = _trace_step(hype.init_step, cfg, state)
    assert seen == [(POP_SIZE, DIM)]
    assert np.array_equal(np.asarray(batch.numpy()), np.asarray(pop_before.numpy()))
    assert np.all(np.isfinite(np.asarray(state.fit.numpy())))
    assert np.allclose(
        np.asarray(state.ref.numpy()),
        np.full(N_OBJS, 1.2 * float(np.max(np.asarray(state.fit.numpy()))), np.float32),
        rtol=1e-6,
    )

    # The first regular step generates its own offspring batch and evaluates
    # THAT batch (not the parents): one evaluate call on (POP_SIZE, DIM)
    # candidates that differ from the parents.
    pop_before = state.pop
    (state, batch), seen = _trace_step(hype.step, cfg, state)
    assert seen == [(POP_SIZE, DIM)]
    assert tuple(batch.shape) == (POP_SIZE, DIM)
    assert not np.array_equal(np.asarray(batch.numpy()), np.asarray(pop_before.numpy()))
    # Fused-step contract: the evaluated batch IS the offspring stored in the
    # state by the generation stage of this same call.
    assert np.array_equal(np.asarray(batch.numpy()), np.asarray(state.offspring.numpy()))


def test_full_run_shapes_and_sanity():
    cfg = make_config()
    state = run_generations(hype, cfg, DTLZ1Config(DIM, N_OBJS), n_gens=N_GENS, seed=SEED)
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
    s1 = run_generations(hype, cfg, DTLZ1Config(DIM, N_OBJS), n_gens=N_GENS, seed=SEED)
    s2 = run_generations(hype, cfg, DTLZ1Config(DIM, N_OBJS), n_gens=N_GENS, seed=SEED)
    assert np.array_equal(np.asarray(s1.pop.numpy()), np.asarray(s2.pop.numpy()))


# --------------------------------------------------------------------------
# Custom crossover / mutation op injection (torch ``HypE.__init__`` parity)
# --------------------------------------------------------------------------


def test_custom_ops_are_invoked():
    """A user-supplied crossover/mutation op is actually called by ``step``.

    The recorders are plain Python functions delegating to the ETL defaults, so
    they fire while ``etl.build`` traces the generation function; each must be
    invoked at least once.
    """
    calls = {"crossover": 0, "mutation": 0}

    def recording_crossover(key, x):
        calls["crossover"] += 1
        return simulated_binary(key, x)

    def recording_mutation(key, x, lb, ub):
        calls["mutation"] += 1
        return polynomial_mutation(key, x, lb, ub)

    cfg = make_config(mutation_op=recording_mutation, crossover_op=recording_crossover)
    assert calls == {"crossover": 0, "mutation": 0}

    # n_gens=2 -> gen 0 is init_step, gen 1 is the single regular step.
    run_generations(hype, cfg, DTLZ1Config(DIM, N_OBJS), n_gens=2, seed=SEED)

    assert calls["crossover"] >= 1
    assert calls["mutation"] >= 1


def test_custom_ops_change_offspring_and_none_matches_default():
    """Identity custom ops change ``state.offspring``; ``None`` op fields
    reproduce the default run bit-for-bit (same RNG draw order/output)."""
    default_state = run_generations(
        hype, make_config(), DTLZ1Config(DIM, N_OBJS), n_gens=2, seed=SEED
    )
    off_default = np.asarray(default_state.offspring.numpy())
    assert off_default.shape == (POP_SIZE, DIM)

    # Identity mutation leaves the (clamped) crossover batch untouched, so the
    # offspring leaf differs from the polynomial-mutation default.
    identity_mut_state = run_generations(
        hype,
        make_config(mutation_op=lambda key, x, lb, ub: x),
        DTLZ1Config(DIM, N_OBJS),
        n_gens=2,
        seed=SEED,
    )
    off_identity_mut = np.asarray(identity_mut_state.offspring.numpy())
    assert off_identity_mut.shape == off_default.shape
    assert not np.array_equal(off_default, off_identity_mut)

    # Identity crossover (no SBX recombination) differs from the default too.
    identity_cross_state = run_generations(
        hype,
        make_config(crossover_op=lambda key, x: x),
        DTLZ1Config(DIM, N_OBJS),
        n_gens=2,
        seed=SEED,
    )
    assert not np.array_equal(
        off_default, np.asarray(identity_cross_state.offspring.numpy())
    )

    # Explicit None == the default: the whole state is reproduced exactly.
    explicit_none = run_generations(
        hype,
        make_config(mutation_op=None, crossover_op=None),
        DTLZ1Config(DIM, N_OBJS),
        n_gens=2,
        seed=SEED,
    )
    assert np.array_equal(
        np.asarray(explicit_none.pop.numpy()), np.asarray(default_state.pop.numpy())
    )
    assert np.array_equal(
        np.asarray(explicit_none.offspring.numpy()), off_default
    )


def test_make_hype_validates_ops_and_hides_selection_op():
    """Non-callable op fields raise ValueError; ``selection_op`` is NOT exposed
    (inert in the torch reference, which hard-codes ``tournament_selection``)."""
    for field in ("mutation_op", "crossover_op"):
        with pytest.raises(ValueError):
            make_config(**{field: 5})
    # Callables and None are both accepted.
    assert make_config(mutation_op=lambda key, x, lb, ub: x).mutation_op is not None
    assert make_config(crossover_op=None).crossover_op is None
    assert not hasattr(hype.HypEConfig, "selection_op")


def test_custom_ops_full_run_sanity():
    """A 3-generation run with both custom ops stays finite and keeps shapes."""
    cfg = make_config(mutation_op=polynomial_mutation, crossover_op=simulated_binary)
    state = run_generations(hype, cfg, DTLZ1Config(DIM, N_OBJS), n_gens=N_GENS, seed=SEED)
    pop = np.asarray(state.pop.numpy())
    fit = np.asarray(state.fit.numpy())
    assert pop.shape == (POP_SIZE, DIM)
    assert fit.shape == (POP_SIZE, N_OBJS)
    assert np.all(np.isfinite(fit))
    assert np.all((pop >= -1e-6) & (pop <= 1.0 + 1e-6))
