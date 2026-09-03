"""Parity test: etl PSO vs the torch evox reference on Sphere (torch allowed)."""

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "unit_test" / "etl" / "algorithms"))

import numpy as np
import torch

from evox.algorithms.so.pso_variants import PSO as TorchPSO
from evox.problems.numerical import Sphere
from evox.workflows import EvalMonitor, StdWorkflow

import evox_etl.algorithms.so.pso_variants.pso as pso_mod
from evox_etl.algorithms.so.pso_variants.pso import PSO
from helpers import SphereConfig, run_generations


def test_pso_parity_with_torch():
    """etl PSO must converge as well as the torch reference on Sphere.

    Uses 100 generations: the two implementations draw different RNG streams
    (etl Threefry keys vs torch MT19937), so after only ~20 generations both
    sit in a high-variance regime (~100 on Sphere dim=40) where paired runs
    differ by up to ~2x.  At 100 generations both converge to single digits,
    where a 1.5x relative bound is a meaningful parity check.
    """
    pop_size, dim, gens = 100, 40, 100

    # torch reference: init_step + 100 steps = 101 fitness evaluations.
    torch.manual_seed(0)
    lb_t = -10 * torch.ones(dim)
    ub_t = 10 * torch.ones(dim)
    monitor = EvalMonitor(full_sol_history=True)
    wf = StdWorkflow(
        TorchPSO(pop_size, lb_t, ub_t), Sphere(), monitor, opt_direction="min"
    )
    wf.init_step()
    for _ in range(gens):
        wf.step()
    # opt_direction="min": topk keeps the smallest fitness (largest=False).
    torch_best = float(monitor.get_topk_fitness()[0])

    # etl port: gens + 1 generations = init eval + gens steps (matches torch).
    cfg = PSO(
        pop_size=pop_size,
        lb=np.full(dim, -10.0, np.float32),
        ub=np.full(dim, 10.0, np.float32),
    )
    state = run_generations(
        pso_mod, cfg, SphereConfig(dim), n_gens=gens + 1, seed=0
    )
    etl_best = float(state.global_best_fit.numpy())

    assert etl_best <= torch_best * 1.5 + 1e-3, (
        f"etl PSO best {etl_best:.6f} vs torch PSO best {torch_best:.6f}"
    )
