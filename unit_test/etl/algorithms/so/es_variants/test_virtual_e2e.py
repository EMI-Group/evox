"""End-to-end convergence tests for the virtual (training-free) ES family.

These tests drive the REAL neuroevolution problem built by
:func:`evox_etl.problems.neuroevolution.make_virtual_problem` (a 4 -> 8 -> 1 ReLU
network on a small fixed synthetic dataset) through :class:`StdWorkflow` with the
real algorithms built by ``make_virtual_es`` / ``make_virtual_lora_es`` — no toy
in-test problem — and assert the network loss actually DECREASES.

The convergence metric is the loss of the pure (unperturbed) center network: the
problem's own ``evaluate`` is re-run with the noise scale pinned to the static
``sigma=0`` (which zeroes the ``sigma * delta`` perturbation), i.e. the exact
network the algorithm carries in its ``(dim,)`` center.
"""

from __future__ import annotations

import numpy as np
import pytest

import etl

from evox_etl.algorithms.so.es_variants import virtual_es, virtual_lora_es
from evox_etl.core.workflow import StdWorkflow
from evox_etl.problems.neuroevolution import virtual_problem
from evox_etl.problems.numerical.state import ProblemState
from evox_etl.workflows import EvalMonitorConfig

# --------------------------------------------------------------------- dataset

#: A tiny 4 -> 8 -> 1 MLP: weight/bias shapes in `named_parameters()` order.
SHAPES = ((8, 4), (8,), (1, 8), (1,))
#: ``("linear", weight_idx, bias_idx, activation)`` per layer.
LAYERS = (("linear", 0, 1, "relu"), ("linear", 2, 3, "identity"))
IN_FEATURES, OUT_FEATURES, BATCH_SIZE = 4, 1, 8
#: Total flat parameter count: 8*4 + 8 + 1*8 + 1.
DIM = 8 * 4 + 8 + 1 * 8 + 1

#: Pop size / step size / noise scale chosen to sit comfortably inside the stable
#: converging regime of the VirtualES gradient estimate (see the noise-budget note
#: in ``es_variants/test_virtual_es.py``).
POP_SIZE = 256
LEARNING_RATE = 0.1
NOISE_STDEV = 0.1
GENERATIONS = 100
LORA_RANK = 2


def _synthetic_dataset() -> tuple[tuple[float, ...], tuple[float, ...]]:
    """A fixed, learnable regression dataset: ``y = x @ w_true`` on 8 samples."""
    rng = np.random.default_rng(0)
    inputs = rng.uniform(-1.0, 1.0, size=(BATCH_SIZE, IN_FEATURES)).astype(np.float32)
    w_true = np.array([1.0, -0.5, 0.25, 0.75], dtype=np.float32)
    targets = (inputs @ w_true).reshape(BATCH_SIZE, OUT_FEATURES).astype(np.float32)
    return tuple(inputs.reshape(-1).tolist()), tuple(targets.reshape(-1).tolist())


INPUTS, TARGETS = _synthetic_dataset()


def _make_problem(lora_rank: int | None = None) -> virtual_problem.VirtualProblemConfig:
    """Build the real virtual neuroevolution problem (full-noise or LoRA mode)."""
    return virtual_problem.make_virtual_problem(
        param_shapes=SHAPES,
        layer_specs=LAYERS,
        inputs=INPUTS,
        targets=TARGETS,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        batch_size=BATCH_SIZE,
        loss="mse",
        reduction="mean",
        lora_rank=lora_rank,
    )


def _center_loss(problem: virtual_problem.VirtualProblemConfig, center: np.ndarray) -> float:
    """Mean squared error of the pure center network (noise scale pinned to 0)."""
    center = np.asarray(center, dtype=np.float32).reshape(DIM)
    seeds = np.zeros(1, dtype=np.int64)

    def body(center_flat, seed_flat):
        # sigma is a STATIC Python float (0.0 kills the `sigma * delta` term), so
        # the forward pass is exactly `center`.
        fitness, _ = virtual_problem.evaluate(problem, ProblemState(), (center_flat, seed_flat, 0.0))
        return fitness

    exe = etl.build(
        body,
        etl.core.TensorSpec((DIM,), np.float32),
        etl.core.TensorSpec((1,), np.int64),
        backend="numpy",
    )
    return float(np.mean(etl.run(exe, center, seeds).numpy()))


def _center_of(state) -> np.ndarray:
    """The returned algorithm ``center`` as a float64 numpy vector."""
    return np.asarray(state.algorithm_state.center.numpy(), dtype=np.float64)


# ------------------------------------------------------------------ VirtualES


def test_virtual_es_end_to_end_convergence():
    """VirtualES + real virtual problem: the center network loss must drop."""
    problem = _make_problem()
    config = virtual_es.make_virtual_es(
        SHAPES,
        pop_size=POP_SIZE,
        center_init=(0.0,) * DIM,
        learning_rate=LEARNING_RATE,
        noise_stdev=NOISE_STDEV,
    )
    workflow = StdWorkflow(config, problem, opt_direction="min")
    state = workflow.run(generations=GENERATIONS, seed=0)

    initial_loss = _center_loss(problem, np.zeros(DIM, np.float32))
    final_loss = _center_loss(problem, _center_of(state))
    best_fitness = float(state.algorithm_state.best_fitness.numpy())

    assert initial_loss > 0.0, "dataset/setup bug: the zero center should incur a nonzero loss"
    assert np.isfinite(final_loss)
    assert final_loss < 0.5 * initial_loss, (
        f"VirtualES did not converge: final center loss {final_loss} vs initial {initial_loss}"
    )
    # The algorithm's own best-so-far fitness (over perturbed draws) must be finite
    # and below the initial-center loss too.
    assert 0.0 <= best_fitness < initial_loss


# --------------------------------------------------------------- VirtualLoRA


def test_virtual_lora_es_end_to_end_convergence():
    """VirtualLoRAES + LoRA-mode virtual problem: the center network loss must drop."""
    problem = _make_problem(lora_rank=LORA_RANK)
    assert problem.lora_rank == LORA_RANK
    config = virtual_lora_es.make_virtual_lora_es(
        SHAPES,
        lora_rank=LORA_RANK,
        pop_size=POP_SIZE,
        center_init=(0.0,) * DIM,
        learning_rate=LEARNING_RATE,
        noise_stdev=NOISE_STDEV,
    )
    workflow = StdWorkflow(config, problem, opt_direction="min")
    state = workflow.run(generations=GENERATIONS, seed=0)

    initial_loss = _center_loss(problem, np.zeros(DIM, np.float32))
    final_loss = _center_loss(problem, _center_of(state))
    best_fitness = float(state.algorithm_state.best_fitness.numpy())

    assert np.isfinite(final_loss)
    assert final_loss < 0.5 * initial_loss, (
        f"VirtualLoRAES did not converge: final center loss {final_loss} vs initial {initial_loss}"
    )
    assert 0.0 <= best_fitness < initial_loss


# ----------------------------------------------------------- monitor hook


def test_virtual_es_workflow_runs_with_eval_monitor():
    """An EvalMonitor-configured VirtualES workflow must trace and run.

    VirtualES hands a ``(center, seeds, sigma)`` payload to ``evaluate``; the
    workflow's optional ``monitor_candidate`` hook maps it onto the ``(pop_size,
    dim)`` tensor the monitor concatenates with its elite buffer.
    """
    problem = _make_problem()
    config = virtual_es.make_virtual_es(
        SHAPES,
        pop_size=64,
        center_init=(0.0,) * DIM,
        learning_rate=LEARNING_RATE,
        noise_stdev=NOISE_STDEV,
    )
    monitor = EvalMonitorConfig(topk=3, full_fit_history=True)
    workflow = StdWorkflow(config, problem, monitor=monitor, opt_direction="min")
    state = workflow.run(generations=12, seed=0)

    assert int(state.generation.numpy()) == 12
    # pop_size/dim are discovered from the algorithm state/config.
    assert workflow.monitor_config.pop_size == 64
    assert workflow.monitor_config.dim == DIM
    best = workflow.monitor.get_best_fitness()
    assert np.isfinite(best)
    assert np.asarray(workflow.monitor.get_latest_solution()).shape == (64, DIM)
    assert len(monitor.fit_history) == 12


@pytest.mark.parametrize("module", [virtual_es, virtual_lora_es])
def test_monitor_candidate_broadcasts_center(module):
    """`monitor_candidate` broadcasts the ``(dim,)`` center to ``(pop, dim)``."""
    dim, pop = 5, 4
    exe = etl.build(
        lambda center, seeds: module.monitor_candidate((center, seeds, 0.1)),
        etl.core.TensorSpec((dim,), np.float32),
        etl.core.TensorSpec((pop,), np.int64),
        backend="numpy",
    )
    out = etl.run(exe, np.arange(dim, dtype=np.float32), np.zeros(pop, np.int64)).numpy()
    assert out.shape == (pop, dim)
    np.testing.assert_array_equal(out, np.tile(np.arange(dim, dtype=np.float32), (pop, 1)))