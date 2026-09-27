"""EvoX HPO with repeated inner evaluation (``num_repeats > 1``) -- runnable example.

This script shows the *repeated-evaluation / fitness-aggregation* path of EvoX's
hyper-parameter optimization (HPO) framework.  An inner ``StdWorkflow`` (PSO on the
Sphere function) is wrapped by ``HPOProblemWrapper``; the outer workflow then tunes
PSO's ``w`` / ``phi_p`` / ``phi_g``.

WHY REPEATS MATTER
------------------
The inner optimisation is *stochastic*: two runs with the same hyper-parameters and
the same initial population, but different random update coefficients, follow
different trajectories and finish with different best fitness values.  The outer
optimiser only ever sees one scalar score per hyper-parameter set, so a single inner
run gives it a noisy target.  Setting ``num_repeats = N`` makes the wrapper run the
inner workflow N times per hyper-parameter row (with different randomness per repeat),
and the inner monitor's MEAN aggregation reduces those N runs to one, less noisy outer
fitness.  The outer optimiser therefore exploits a smoother estimate of "how good is
this hyper-parameter set".

Run (CPU-only is fine, exits in a few seconds):
    /home/bill/Source/evox/.venv/bin/python examples/hpo/repeated_hpo.py
"""

import pathlib
import sys
import time

# Make `evox` importable without relying on the editable install: walk up from this
# file until we find the repo root that contains `src/evox/__init__.py`.
_here = pathlib.Path(__file__).resolve()
for _p in _here.parents:
    if (_p / "src" / "evox" / "__init__.py").is_file():
        _ROOT = _p
        break
else:  # pragma: no cover
    raise RuntimeError("could not locate the repo root (src/evox/__init__.py)")
for _path in (str(_ROOT), str(_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import torch

from evox.algorithms import PSO
from evox.core import vmap
from evox.problems.hpo_wrapper import (
    HPOFitnessMonitor,
    HPOProblemWrapper,
    _hpo_evaluate_loop,
    get_sub_state,
)
from evox.problems.numerical import Sphere
from evox.workflows import EvalMonitor, StdWorkflow

# --- configuration (kept tiny: repeated evaluation multiplies the inner work) -----
DIM = 5  # dimension of the Sphere problem solved by the inner PSO
INNER_POP = 10  # inner PSO population size
INNER_ITERATIONS = 12  # inner generations per repeat
NUM_INSTANCES = 5  # outer population size == number of hyper-parameter rows
NUM_REPEATS = 3  # inner runs per hyper-parameter row  <-- the feature on show
OUTER_GENERATIONS = 6  # short outer loop (this script is about aggregation, not long search)
BOUND = 10.0  # inner search box [-BOUND, BOUND]^DIM
BUILD_SEED = 0  # seeds the workflows' construction (their initial populations/parameters)
EVAL_SEED = 3  # seeds each evaluate call so every printed number is reproducible


def _no_aggregation(fitness: torch.Tensor) -> torch.Tensor:
    """``fit_aggregation`` that keeps every repeat separate (identity, no averaging).

    Used only to make the individual per-repeat inner runs *visible*: with the
    default MEAN aggregation the repeat axis is averaged away, so all repeats end up
    reporting the same aggregated value.
    """
    return fitness


def _raw_repeat_fitness(hpo_prob: HPOProblemWrapper, params: dict) -> torch.Tensor:
    """Return the wrapper's internal ``fit`` tensor with the repeat axis visible.

    ``HPOProblemWrapper.evaluate`` runs the vmapped inner workflow and returns
    ``fit[0]`` of ``vmap(torch.vmap(monitor.tell_fitness))(monitor_state)`` -- i.e. it
    slices the leading repeat axis away.  This helper performs exactly the same
    computation (same private attributes and the same ``_hpo_evaluate_loop`` op that
    ``evaluate`` uses) but returns the *full* ``fit`` tensor of shape
    ``(num_repeats, num_instances)`` so the repeat axis can be inspected.  It does not
    mutate the wrapper (the loop op works on cloned state).
    """
    if hpo_prob.num_repeats <= 1:
        raise ValueError("this helper only makes sense for num_repeats > 1")

    state_params = {**hpo_prob._init_params, **params}
    buffers = {k: v.clone() for k, v in hpo_prob._init_buffers.items()}
    state_params, buffers = hpo_prob._workflow_init_step_(state_params, buffers)
    values = [state_params[k] for k in hpo_prob.state_keys[0]] + [buffers[k] for k in hpo_prob.state_keys[1]]
    values = _hpo_evaluate_loop(False, hpo_prob._id_, hpo_prob.iterations - 2, values)
    state_params = {k: v for k, v in zip(hpo_prob.state_keys[0], values)}
    buffers = {k: v for k, v in zip(hpo_prob.state_keys[1], values[len(state_params) :])}
    state_params, buffers = hpo_prob._workflow_final_step_(state_params, buffers)
    monitor_state = get_sub_state(buffers, "monitor")
    _, fit = vmap(torch.vmap(hpo_prob._stateful_tell_fitness))(monitor_state)
    return fit


def _build_hpo_problem(fit_aggregation=None) -> tuple[HPOProblemWrapper, StdWorkflow]:
    """Build the HPO problem wrapper around a fresh PSO-on-Sphere inner workflow."""
    lb = -BOUND * torch.ones(DIM)
    ub = BOUND * torch.ones(DIM)
    if fit_aggregation is None:
        monitor = HPOFitnessMonitor(num_repeats=NUM_REPEATS)  # default MEAN aggregation
    else:
        monitor = HPOFitnessMonitor(num_repeats=NUM_REPEATS, fit_aggregation=fit_aggregation)
    inner_workflow = StdWorkflow(
        algorithm=PSO(pop_size=INNER_POP, lb=lb, ub=ub),
        problem=Sphere(),
        monitor=monitor,
        opt_direction="min",
    )
    hpo_prob = HPOProblemWrapper(
        iterations=INNER_ITERATIONS,
        num_instances=NUM_INSTANCES,
        workflow=inner_workflow,
        num_repeats=NUM_REPEATS,
        copy_init_state=True,
    )
    return hpo_prob, inner_workflow


def _make_params(hpo_prob: HPOProblemWrapper, seed: int = 0) -> dict:
    """One distinct hyper-parameter row per outer instance (shape ``(num_instances, 1)``)."""
    torch.manual_seed(seed)
    params = hpo_prob.get_init_params()
    for key in list(params):
        params[key] = torch.nn.Parameter(torch.rand(NUM_INSTANCES, 1) * 2.0 + 0.5, requires_grad=False)
    return params


def main() -> None:
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_printoptions(precision=4, sci_mode=False)
    t_start = time.perf_counter()

    print("=" * 78)
    print("EvoX HPO with repeated inner evaluation (num_repeats > 1)")
    print(
        f"device={torch.get_default_device()}  inner=PSO(pop={INNER_POP}) on Sphere(dim={DIM})  iterations={INNER_ITERATIONS}"
    )
    print(f"outer instances={NUM_INSTANCES}  num_repeats={NUM_REPEATS}")
    print("=" * 78)

    # --- Stage 1: inner workflow + HPO wrapper ----------------------------------
    # The inner StdWorkflow minimises Sphere; HPOFitnessMonitor tracks its best
    # fitness and (with num_repeats>1) aggregates the repeats with the default MEAN.
    torch.manual_seed(BUILD_SEED)
    hpo_prob, inner_workflow = _build_hpo_problem()
    print("\n[1] Tunable hyper-parameter keys:", hpo_prob.get_params_keys())
    print(
        "    inner monitor fit_aggregation:",
        inner_workflow.monitor.fit_aggregation,
        "-> MEAN over the repeat axis",
    )

    # --- Stage 2: distinct hyper-parameter rows ---------------------------------
    params = _make_params(hpo_prob, seed=BUILD_SEED)
    print("\n[2] Hyper-parameter rows (one row = one outer candidate):")
    for key, value in params.items():
        print(f"    {key:<20} shape={tuple(value.shape)} values={value.detach().reshape(-1)}")

    # --- Stage 3: aggregated outer evaluation -----------------------------------
    # A single call to `evaluate` runs ALL instances x ALL repeats, then returns the
    # aggregated outer fitness (one value per hyper-parameter row).
    t0 = time.perf_counter()
    torch.manual_seed(EVAL_SEED)
    aggregated = hpo_prob.evaluate(params)
    print("\n[3] hpo_prob.evaluate(params)  -- aggregated outer fitness")
    print(f"    shape={tuple(aggregated.shape)}  dtype={aggregated.dtype}")
    print(f"    values={aggregated}    ({time.perf_counter() - t0:.3f}s)")
    # one aggregated scalar per hyper-parameter row (the repeat axis is collapsed)
    assert aggregated.shape == (NUM_INSTANCES,), aggregated.shape

    # --- Stage 4: expose the repeat axis ----------------------------------------
    # `evaluate` swallowed `fit[0]`; recover the full (num_repeats, num_instances)
    # tensor to show the repeat dimension explicitly. Same seed -> same draws, so
    # `raw[0]` must equal the value `evaluate` returned in stage 3.
    torch.manual_seed(EVAL_SEED)
    raw = _raw_repeat_fitness(hpo_prob, params)
    assert raw.shape == (NUM_REPEATS, NUM_INSTANCES), raw.shape
    print("\n[4] Raw fitness tensor WITH the repeat axis (default MEAN aggregation)")
    print(f"    shape={tuple(raw.shape)}  (dim 0 = num_repeats, dim 1 = num_instances)")
    print(f"{raw}")
    print("    mean over the repeat axis:", raw.mean(dim=0))
    print(f"    evaluate(...) == raw[0] : {torch.equal(aggregated, raw[0])}")
    print(
        "    -> every repeat row is identical: the MEAN aggregation averages the\n"
        "       repeat axis inside the vmapped step, so each repeat reports the SAME\n"
        "       aggregated value and `evaluate` returning `fit[0]` is that aggregate."
    )

    # --- Stage 5: the N individual inner runs (no aggregation) -------------------
    # Build a sibling wrapper with an identity `fit_aggregation` so the repeat axis
    # is NOT collapsed: the rows now hold the N genuinely independent inner runs,
    # which is what the MEAN aggregation in stages [3]/[4] consumes.
    torch.manual_seed(BUILD_SEED)
    hpo_raw, _ = _build_hpo_problem(fit_aggregation=_no_aggregation)
    params_raw = {k: torch.nn.Parameter(v.detach().clone(), requires_grad=False) for k, v in params.items()}
    torch.manual_seed(EVAL_SEED)
    raw_runs = _raw_repeat_fitness(hpo_raw, params_raw)
    assert raw_runs.shape == (NUM_REPEATS, NUM_INSTANCES), raw_runs.shape
    print("\n[5] The N independent inner runs (identity aggregation, repeats kept)")
    print(f"    shape={tuple(raw_runs.shape)}  (rows = repeats, columns = instances)")
    print(f"{raw_runs}")
    print("    mean over the repeat axis:", raw_runs.mean(dim=0))
    print(
        "    -> the rows differ: each repeat is a different stochastic PSO run of the\n"
        "       same hyper-parameters. The MEAN aggregation used in [3]/[4] reduces the\n"
        "       repeat axis generation by generation (inside the vmap), so its value is\n"
        "       not identical to the plain mean of these final per-repeat bests -- but\n"
        "       both are ways of turning N noisy runs into one less noisy score."
    )

    # --- Stage 6: a short outer optimisation loop --------------------------------
    # Outer PSO searches (w, phi_p, phi_g) in [0, 3]^3; the outer population size
    # must equal `num_instances` and `solution_transform` maps a solution to the
    # tunable keys. HPO exists only in `src/evox` (the torch reference).
    class solution_transform(torch.nn.Module):
        def forward(self, x: torch.Tensor):
            return {
                "algorithm.w": x[:, 0],
                "algorithm.phi_p": x[:, 1],
                "algorithm.phi_g": x[:, 2],
            }

    torch.manual_seed(BUILD_SEED)
    outer_algo = PSO(pop_size=NUM_INSTANCES, lb=0 * torch.ones(3), ub=3 * torch.ones(3))
    outer_monitor = EvalMonitor()
    outer_workflow = StdWorkflow(
        algorithm=outer_algo,
        problem=hpo_prob,
        monitor=outer_monitor,
        solution_transform=solution_transform(),
    )

    print(f"\n[6] Outer PSO over (w, phi_p, phi_g), {OUTER_GENERATIONS} generations")
    torch.manual_seed(EVAL_SEED)
    outer_workflow.init_step()
    print(f"    generation 0 (init): aggregated best = {float(outer_monitor.get_best_fitness()):.4f}")
    for generation in range(1, OUTER_GENERATIONS + 1):
        outer_workflow.step()
        best = float(outer_monitor.get_best_fitness())
        print(f"    generation {generation}: aggregated best = {best:.4f}")

    best_hyper = outer_monitor.get_best_solution()
    print("\n    Final best hyper-parameters (w, phi_p, phi_g):")
    print(f"      w={float(best_hyper[0]):.4f}  phi_p={float(best_hyper[1]):.4f}  phi_g={float(best_hyper[2]):.4f}")

    # --- Stage 7: what exactly is being averaged --------------------------------
    print("\n[7] What is being averaged?")
    print(
        "    * Each outer candidate is one row of hyper-parameters. Its inner\n"
        "      optimisation is stochastic, so a single run gives a noisy score.\n"
        f"    * num_repeats={NUM_REPEATS} stacks a repeat axis (size {NUM_REPEATS}) onto every inner\n"
        "      state and runs the inner workflow once per repeat with different\n"
        "      randomness (nested vmap: outer = repeats 'different', inner = instances\n"
        "      'same', so repeats diverge while instances share a draw).\n"
        "    * The inner monitor's default fit_aggregation (_mean_fit_aggregation,\n"
        "      MEAN) averages the fitness across that repeat axis inside the vmapped\n"
        "      step, collapsing the N noisy inner results into ONE outer fitness per\n"
        "      hyper-parameter row -- exactly the scalar the outer optimiser scores.\n"
        "    * `evaluate` then returns fit[0] of the (num_repeats, num_instances)\n"
        "      tensor, hence the shape (num_instances,) printed in step [3].\n"
        "    * Repeats matter because the inner search is random: averaging N runs\n"
        "      gives the outer optimiser a smoother, less noisy target."
    )
    print(f"\nDone in {time.perf_counter() - t_start:.2f}s.")


if __name__ == "__main__":
    main()
