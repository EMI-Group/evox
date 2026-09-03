"""Parity test: torch evox DE vs functional evox_etl de on Sphere (torch allowed)."""
import sys
from pathlib import Path
sys.path[0:0] = [str(Path(__file__).resolve().parents[4]), str(Path(__file__).resolve().parents[4] / "src")]
import numpy as np
import torch
from evox.algorithms import DE as TorchDE
from evox.problems.numerical import basic
from evox.workflows import EvalMonitor, StdWorkflow
import evox_etl.algorithms.so.de_variants.de as de_mod
from evox_etl.algorithms.so.de_variants import DE
from unit_test.etl.algorithms.helpers import SphereConfig, run_generations

POP, DIM, GENS, SEED = 100, 40, 20, 0

def test_de_parity_sphere():
    torch.manual_seed(SEED)
    algo = TorchDE(POP, torch.full((DIM,), -100.0), torch.full((DIM,), 100.0))
    prob = basic.Sphere()
    monitor = EvalMonitor(full_sol_history=True)
    wf = StdWorkflow(algo, prob, monitor=monitor)
    wf.init_step()
    for _ in range(GENS):
        wf.step()
    torch_best = float(monitor.get_best_fitness())
    cfg = DE(pop_size=POP, lb=np.full(DIM, -100.0), ub=np.full(DIM, 100.0))
    state = run_generations(de_mod, cfg, SphereConfig(DIM), GENS, seed=SEED)
    etl_best = float(np.min(state.fit.numpy()))
    assert etl_best <= torch_best * 1.1 + 1e-3, f"etl best {etl_best} > torch best {torch_best} * 1.1 + 1e-3"
