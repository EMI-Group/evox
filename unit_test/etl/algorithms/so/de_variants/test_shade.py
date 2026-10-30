"""Smoke test for the functional SHADE port (no torch)."""
import sys
from pathlib import Path

sys.path[0:0] = [str(Path(__file__).resolve().parents[5]), str(Path(__file__).resolve().parents[5] / "src")]

import numpy as np

import evox_etl.algorithms.so.de_variants.shade as shade_mod
from unit_test.etl.algorithms.helpers import SphereConfig, run_generations

POP, DIM, GENS = 10, 20, 3


def test_shade_smoke():
    cfg = shade_mod.make_shade(pop_size=POP, lb=np.full(DIM, -10.0), ub=np.full(DIM, 10.0))
    state = run_generations(shade_mod, cfg, SphereConfig(DIM), GENS, seed=0)
    fit = state.fit.numpy()
    assert fit.shape == (POP,) and np.all(np.isfinite(fit))
    assert state.pop.numpy().shape == (POP, DIM)
    mem = state.Memory_FCR.numpy()
    assert mem.shape == (2, POP) and np.all(np.isfinite(mem))


# --- success-history adaptation (regression: memories used to stay frozen) ----
ADAPT_POP, ADAPT_DIM, ADAPT_GENS = 20, 10, 25


def test_shade_memory_adapts_and_converges():
    """The F/CR success-history memory must move off its 0.5 initialization
    while the run still converges on Sphere.

    The memory starts at ``0.5`` everywhere; a port that never writes the
    adapting entries keeps it bit-identical at ``0.5``, so any entry drifting
    by more than a couple of percent proves the success-history update runs.
    Thresholds are deliberately loose (observed drift ~0.09-0.15, best fitness
    ~0.13-0.28 over seeds 0/1/2/42) so the test is not seed- or noise-flaky.
    """
    cfg = shade_mod.make_shade(
        pop_size=ADAPT_POP, lb=np.full(ADAPT_DIM, -5.0), ub=np.full(ADAPT_DIM, 5.0)
    )
    state = run_generations(shade_mod, cfg, SphereConfig(ADAPT_DIM), ADAPT_GENS, seed=0)
    mem = state.Memory_FCR.numpy()
    assert mem.shape == (2, ADAPT_POP) and np.all(np.isfinite(mem))
    assert not np.all(mem == np.float32(0.5))
    assert np.abs(mem - 0.5).max() > 0.02
    # still converges (Sphere is 10-D here, starting from [-5, 5]^10)
    fit = state.fit.numpy()
    assert fit.shape == (ADAPT_POP,) and np.all(np.isfinite(fit))
    assert fit.min() < 5.0
