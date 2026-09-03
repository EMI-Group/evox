"""Smoke test for the functional RVEA port (``evox_etl.algorithms.mo.rvea``).

ETL-only, no torch. Checks the gen-0 contract (``init_ask`` evaluates the FULL
initial population, later ``ask`` draws the offspring batch), a 3-generation
DTLZ1 run via ``helpers.run_generations``, and seed determinism.
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

import helpers
from evox_etl.algorithms.mo import rvea
from helpers import DTLZ1Config, run_generations

POP_SIZE, N_OBJS, DIM, N_GENS, SEED = 20, 3, 7, 3, 0
# Effective population: uniform_sampling overwrites pop_size=20 with the
# Das-Dennis count for 3 objectives (15) — same as the torch reference.
N_EFF = 15
# Ask draws an (N_EFF, dim) mating pool, then SBX pairs rows n//2 and returns
# 2*(n//2) children -> (14, 7). Same quirk as the torch RVEA.
N_OFFSPRING = 2 * (N_EFF // 2)


def make_config():
    """RVEA config: 3 objectives, decision space [0, 1]^7."""
    return rvea.RVEAConfig(
        pop_size=POP_SIZE,
        n_objs=N_OBJS,
        lb=np.zeros(DIM, dtype=np.float32),
        ub=np.ones(DIM, dtype=np.float32),
    )


def _leaf_spec(t):
    """TensorSpec mirroring one etl tensor leaf."""
    return etl.core.TensorSpec(shape=tuple(t.shape), dtype=np.dtype(t.dtype))


def _spec_of(a):
    """TensorSpec for a tensor leaf, or a spec pytree for a state dataclass."""
    return _leaf_spec(a) if hasattr(a, "numpy") else etl.tree_map(_leaf_spec, a)


def _run(fn, cfg, *args):
    """Build ``fn(cfg, *args)`` with TensorSpec mirrors and run it once."""
    exe = etl.build(fn, cfg, *[_spec_of(a) for a in args], backend="numpy")
    return etl.run(exe, cfg, *args)


def _evaluate(candidates):
    """One DTLZ1 evaluation of a candidate batch (toy problems are stateless)."""
    cfg = DTLZ1Config(DIM, N_OBJS)
    exe = etl.build(
        helpers.toy_evaluate, cfg, helpers.ToyProblemState(),
        _leaf_spec(candidates), backend="numpy",
    )
    fitness, _ = etl.run(exe, cfg, helpers.ToyProblemState(), candidates)
    return fitness


def test_init_ask_full_population_and_ask_offspring_shape():
    cfg = make_config()
    key_spec = etl.core.TensorSpec(shape=(), dtype=np.dtype("int64"))
    init_exe = etl.build(rvea.init, cfg, key_spec, backend="numpy")
    state = etl.run(init_exe, cfg, np.asarray(SEED, dtype=np.int64))

    # Gen 0 evaluates the FULL initial population (N_EFF rows), not an
    # offspring batch.
    candidates, state = _run(rvea.init_ask, cfg, state)
    assert tuple(candidates.shape) == tuple(state.pop.shape) == (N_EFF, DIM)

    # Record the gen-0 fitness, then the first regular ask draws the offspring.
    state = _run(rvea.init_tell, cfg, state, _evaluate(candidates))
    offspring, _ = _run(rvea.ask, cfg, state)
    assert tuple(offspring.shape) == (N_OFFSPRING, DIM)


def test_full_run_shapes_and_sanity():
    cfg = make_config()
    state = run_generations(rvea, cfg, DTLZ1Config(DIM, N_OBJS), n_gens=N_GENS, seed=SEED)
    pop = np.asarray(state.pop.numpy())
    fit = np.asarray(state.fit.numpy())
    assert pop.shape == (N_EFF, DIM)
    assert fit.shape == (N_EFF, N_OBJS)
    # Torch-faithful: RVEA selection keeps one survivor row per reference
    # vector, NaN-filled for unmatched vectors (verified against the torch
    # ref_vec_guided op), so the fit may contain NaN rows.
    assert np.isfinite(np.nanmin(fit))
    # DTLZ1 optimum is 0; after 3 gens each objective's best is well below
    # 100 (loose sanity bound, NaN-ignoring).
    assert np.all(np.nanmin(fit, axis=0) < 100.0)
    assert np.nanmin(pop) >= -1e-6 and np.nanmax(pop) <= 1.0 + 1e-6


def test_full_run_deterministic():
    cfg = make_config()
    s1 = run_generations(rvea, cfg, DTLZ1Config(DIM, N_OBJS), n_gens=N_GENS, seed=SEED)
    s2 = run_generations(rvea, cfg, DTLZ1Config(DIM, N_OBJS), n_gens=N_GENS, seed=SEED)
    # equal_nan: NaN != NaN in plain array_equal, and NaN survivor rows exist.
    assert np.allclose(
        np.asarray(s1.pop.numpy()), np.asarray(s2.pop.numpy()),
        rtol=0.0, atol=0.0, equal_nan=True,
    )
