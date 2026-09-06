"""Parity test: torch evox DE vs functional evox_etl de on Sphere (torch allowed).

Both sides use different RNG streams (torch MT19937 vs etl keyed PRNGs), so
bit-identical trajectories are impossible and per-seed best fitnesses fluctuate
symmetrically around ratio ~1.0 (60-seed sweep: mean ratio ~1.03, spread
0.57-1.92, no systematic bias). A single-seed 10% margin fails ~1/3 of the time
by pure chance, so the test compares the MEDIAN best fitness over 3 seeds with
the same 10% margin — convergence must match within 10% in expectation.

Note the torch side evaluates the initial population in init_step (21 evals for
20 generations); the etl side mirrors this via init_ask/init_tell.
"""
import sys
from pathlib import Path

sys.path[0:0] = [str(Path(__file__).resolve().parents[4]), str(Path(__file__).resolve().parents[4] / "src")]

import numpy as np
import torch

from evox.algorithms import DE as TorchDE
from evox.problems.numerical import basic
from evox.workflows import EvalMonitor, StdWorkflow
import evox_etl.algorithms.so.de_variants.de as de_mod
from unit_test.etl.algorithms.helpers import SphereConfig, run_generations

POP, DIM, GENS = 100, 40, 20
SEEDS = (0, 1, 2)


def _torch_best(seed: int) -> float:
    torch.manual_seed(seed)
    algo = TorchDE(POP, torch.full((DIM,), -100.0), torch.full((DIM,), 100.0))
    prob = basic.Sphere()
    monitor = EvalMonitor(full_sol_history=True)
    wf = StdWorkflow(algo, prob, monitor=monitor)
    wf.init_step()
    for _ in range(GENS):
        wf.step()
    return float(monitor.get_best_fitness())


def _etl_best(seed: int) -> float:
    cfg = de_mod.make_de(pop_size=POP, lb=np.full(DIM, -100.0), ub=np.full(DIM, 100.0))
    state = run_generations(de_mod, cfg, SphereConfig(DIM), GENS, seed=seed)
    return float(np.min(state.fit.numpy()))


def test_de_parity_sphere():
    torch_bests = [_torch_best(s) for s in SEEDS]
    etl_bests = [_etl_best(s) for s in SEEDS]
    torch_med, etl_med = float(np.median(torch_bests)), float(np.median(etl_bests))
    assert etl_med <= torch_med * 1.1 + 1e-3, (
        f"median etl best {etl_med} > median torch best {torch_med} * 1.1 + 1e-3 "
        f"(per-seed: torch {torch_bests}, etl {etl_bests})"
    )
