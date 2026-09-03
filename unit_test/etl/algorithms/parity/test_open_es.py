"""Parity tests: the functional ETL OpenES port vs the torch evox reference.

Both sides minimize the 40-dim sphere f(x) = sum(x**2) from the same
deterministic initial center and run the SAME number of generations.
The torch side drives `OpenES` through `StdWorkflow` + `EvalMonitor`;
`OpenES` does not override `init_step`, so `init_step()` falls back to a
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

import evox_etl.algorithms.so.es_variants.open_es as etl_open
from evox.algorithms import OpenES
from evox.problems.numerical import Sphere
from evox.workflows import EvalMonitor, StdWorkflow

DIM = 40
# Shared deterministic initial center, identical on both sides.
CENTER = np.random.default_rng(123).uniform(-5.0, 5.0, DIM).astype(np.float32)
INITIAL_FITNESS = float(np.sum(CENTER.astype(np.float64) ** 2))  # ~342
N_GENS = 200
POP_SIZE = 64
LEARNING_RATE = 0.01
NOISE_STDEV = 2.0


def _torch_best_fitness(seed: int, optimizer: str | None) -> tuple[float, int]:
    """Run the torch OpenES for N_GENS generations; return (best fitness, n_gens)."""
    torch.manual_seed(seed)
    algorithm = OpenES(
        pop_size=POP_SIZE,
        center_init=torch.from_numpy(CENTER),
        learning_rate=LEARNING_RATE,
        noise_stdev=NOISE_STDEV,
        optimizer=optimizer,
        mirrored_sampling=True,
    )
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
@pytest.mark.parametrize("optimizer", [None, "adam"])
def test_open_es_parity(seed: int, optimizer: str | None):
    torch_best, n_gens = _torch_best_fitness(seed, optimizer)
    config = etl_open.OpenESConfig(
        pop_size=POP_SIZE,
        center_init=CENTER,
        learning_rate=LEARNING_RATE,
        noise_stdev=NOISE_STDEV,
        optimizer=optimizer,
        mirrored_sampling=True,
    )
    state = run_generations(etl_open, config, SphereConfig(DIM), n_gens, seed=seed)
    etl_best = float(state.best_fitness.numpy())

    assert np.isfinite(torch_best) and np.isfinite(etl_best)
    # The torch side must actually improve over the initial center fitness,
    # otherwise the parity comparison would be meaningless.
    assert torch_best < 0.5 * INITIAL_FITNESS
    assert etl_best <= torch_best * 1.1 + 1e-3
