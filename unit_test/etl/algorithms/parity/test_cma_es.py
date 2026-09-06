"""Parity test: the functional ETL CMA-ES port vs the torch evox reference.

Both sides minimize the 40-dim sphere f(x) = sum(x**2) from the same
deterministic initial mean and run the SAME number of generations.
The torch side drives `CMAES` through `StdWorkflow` + `EvalMonitor`;
`CMAES` does not override `init_step`, so `init_step()` falls back to a
plain `step()` and 1 init_step + (N_GENS - 1) steps == N_GENS generations
(verified by counting `monitor.fitness_history` entries, which the etl
side's `n_gens` is aligned to).

The RNG streams differ (torch global `manual_seed` vs the etl keyed RNG),
so only convergence parity within 10% is asserted: the etl port may be
better, but must not be worse by more than the margin.
"""

import numpy as np
import pytest
import torch

import helpers
from helpers import SphereConfig, run_generations

import evox_etl.algorithms.so.es_variants.cma_es as etl_cma
from evox.algorithms import CMAES
from evox.problems.numerical import Sphere
from evox.workflows import EvalMonitor, StdWorkflow

DIM = 40
# Shared deterministic initial mean, identical on both sides.
CENTER = np.random.default_rng(123).uniform(-5.0, 5.0, DIM).astype(np.float32)
INITIAL_FITNESS = float(np.sum(CENTER.astype(np.float64) ** 2))  # ~342
N_GENS = 80
SIGMA = 5.0


def _torch_best_fitness(seed: int) -> tuple[float, int]:
    """Run the torch CMAES for N_GENS generations; return (best fitness, n_gens)."""
    torch.manual_seed(seed)
    algorithm = CMAES(mean_init=torch.from_numpy(CENTER), sigma=SIGMA)
    problem = Sphere()  # dim is inferred from the algorithm's population (40)
    monitor = EvalMonitor(full_sol_history=True)
    workflow = StdWorkflow(algorithm, problem, monitor=monitor)
    workflow.init_step()
    for _ in range(N_GENS - 1):
        workflow.step()
    n_gens = len(monitor.fitness_history)
    assert n_gens == N_GENS, "expected init_step + steps to equal N_GENS generations"
    return float(monitor.get_best_fitness()), n_gens


@pytest.mark.parametrize("seed", [0, 1])
def test_cma_es_parity(seed: int):
    torch_best, n_gens = _torch_best_fitness(seed)
    config = etl_cma.make_cma_es(mean_init=CENTER, sigma=SIGMA)
    state = run_generations(etl_cma, config, SphereConfig(DIM), n_gens, seed=seed)
    etl_best = float(state.best_fitness.numpy())

    assert np.isfinite(torch_best) and np.isfinite(etl_best)
    # The torch side must actually improve over the initial center fitness,
    # otherwise the parity comparison would be meaningless.
    assert torch_best < 0.5 * INITIAL_FITNESS
    assert etl_best <= torch_best * 1.1 + 1e-3
