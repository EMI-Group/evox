"""Parity test: evox_etl NSGA2 vs the torch evox reference on DTLZ1.

Setting (both sides, equivalent): dim=7, n_obj=3, pop_size=100, seed=42,
20 offspring generations, DTLZ1 (optimum = 0 per objective).

- torch side: ``StdWorkflow`` + ``EvalMonitor``; best = per-objective
  minimum over the whole-history Pareto front.
- etl side: the functional init/ask/tell port driven by
  ``helpers.run_generations`` on the numpy backend; best = per-objective
  minimum of the final population (NSGA-II is elitist — the final
  population retains the front — so final-pop min == history min).

The two sides draw different RNG streams, so exact trajectories differ; the
assertion checks convergence parity with a 10% relative margin and a 0.05
absolute floor (DTLZ1's optimum is 0 per objective).
"""
import numpy as np

import evox
from evox_etl.algorithms.mo import nsga2
from unit_test.etl.algorithms.helpers import DTLZ1Config, run_generations

from unit_test.etl.algorithms.parity.parity_common import ABS_TOL, DIM, N_GENS, N_OBJ, POP_SIZE, REL_MARGIN, SEED, torch_reference


def test_nsga2_dtlz1_parity():
    cfg = nsga2.NSGA2Config(
        pop_size=POP_SIZE,
        n_objs=N_OBJ,
        lb=np.zeros(DIM, np.float32),
        ub=np.ones(DIM, np.float32),
    )
    state = run_generations(nsga2, cfg, DTLZ1Config(DIM, N_OBJ), n_gens=N_GENS, seed=SEED)
    etl_best = np.min(np.asarray(state.fit.numpy()), axis=0)

    torch_best, torch_final = torch_reference(evox.algorithms.NSGA2)

    print(f"NSGA2 parity (DTLZ1 d={DIM} m={N_OBJ}, {N_GENS} gens, seed {SEED}):")
    print(f"  etl   final-pop min = {etl_best}")
    print(f"  torch PF-hist min   = {torch_best}")
    print(f"  torch final-pop min = {torch_final}")

    for k in range(N_OBJ):
        bound = torch_best[k] * REL_MARGIN + ABS_TOL
        assert etl_best[k] <= bound, (
            f"objective {k}: etl {etl_best[k]} worse than torch {torch_best[k]} "
            f"beyond the {ABS_TOL}+{REL_MARGIN}x margin (bound {bound})"
        )
