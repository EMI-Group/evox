"""EvoX HPO with repeated inner evaluation (``num_repeats > 1``) -- runnable example.

Uses ONLY the public EvoX API to demonstrate the *repeated-evaluation /
fitness-aggregation* path of hyper-parameter optimization (HPO).  An inner
``StdWorkflow`` (PSO on Sphere) is wrapped by ``HPOProblemWrapper``; a short outer
PSO then tunes PSO's ``w`` / ``phi_p`` / ``phi_g``.

WHY REPEATS MATTER: the inner optimisation is stochastic, so a single run gives the
outer optimiser a noisy score.  With ``num_repeats = N`` the wrapper runs the inner
workflow N times per hyper-parameter row and the monitor's default MEAN aggregation
(``HPOFitnessMonitor.fit_aggregation``) collapses those N runs into ONE outer fitness
-- ``evaluate`` still returns a single scalar per row.  Averaging N stochastic runs is
the intended way to give the outer optimiser a smoother target; this script also
*measures*, with public calls only, how the per-row score spread actually behaves.

Run (CPU-only is fine, exits in a few seconds):
    /home/bill/Source/evox/.venv/bin/python examples/hpo/repeated_hpo.py
"""

import pathlib
import sys
import time

# Make `evox` importable without relying on the editable install: find the repo root.
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
from evox.problems.hpo_wrapper import HPOFitnessMonitor, HPOProblemWrapper
from evox.problems.numerical import Sphere
from evox.workflows import EvalMonitor, StdWorkflow

# --- configuration (kept tiny: repeated evaluation multiplies the inner work) -----
DIM = 5  # dimension of the Sphere problem solved by the inner PSO
INNER_POP = 10  # inner PSO population size
INNER_ITERATIONS = 10  # inner generations per repeat
NUM_INSTANCES = 5  # outer population size == number of hyper-parameter rows
NUM_REPEATS = 3  # inner runs per hyper-parameter row  <-- the feature on show
OUTER_GENERATIONS = 5  # short outer loop (this script is about aggregation, not long search)
NUM_EVAL_CALLS = 5  # independent evaluate() calls used to measure the score spread
BOUND = 10.0  # inner search box [-BOUND, BOUND]^DIM
BUILD_SEED = 0  # seed for constructing the workflows (their initial populations)
EVAL_SEED = 3  # seed for the evaluate calls, so every printed number is reproducible
# Per-key perturbation ranges for the tuned rows (small enough to keep the inner PSO
# well behaved; all sit inside the outer search box [0, 3] used in stage [5]).
PARAM_RANGES = {"algorithm.w": (0.2, 0.9), "algorithm.phi_p": (1.0, 2.5), "algorithm.phi_g": (0.5, 1.5)}


def _build_hpo_problem(num_repeats: int) -> tuple[HPOProblemWrapper, StdWorkflow]:
    """Wrap a fresh PSO-on-Sphere inner workflow in an HPO problem.

    Monitor and wrapper get the SAME ``num_repeats`` (the documented pattern).
    """
    lb, ub = -BOUND * torch.ones(DIM), BOUND * torch.ones(DIM)
    monitor = HPOFitnessMonitor(num_repeats=num_repeats)  # default MEAN aggregation
    inner_workflow = StdWorkflow(
        algorithm=PSO(pop_size=INNER_POP, lb=lb, ub=ub), problem=Sphere(), monitor=monitor, opt_direction="min"
    )
    hpo_prob = HPOProblemWrapper(
        iterations=INNER_ITERATIONS,
        num_instances=NUM_INSTANCES,
        workflow=inner_workflow,
        num_repeats=num_repeats,
        copy_init_state=True,
    )
    return hpo_prob, inner_workflow


def _make_params(hpo_prob: HPOProblemWrapper, seed: int) -> dict:
    """Distinct hyper-parameter rows (shape ``(num_instances, 1)``) per instance.

    Follows the ``HPOProblemWrapper`` docstring: start from ``get_init_params()`` and
    overwrite the entries we want to tune.
    """
    torch.manual_seed(seed)
    params = hpo_prob.get_init_params()
    for key in list(params):
        lo, hi = PARAM_RANGES[key]
        params[key] = torch.nn.Parameter(torch.rand(NUM_INSTANCES, 1) * (hi - lo) + lo, requires_grad=False)
    return params


def _sample(hpo_prob: HPOProblemWrapper, params: dict, seeds) -> torch.Tensor:
    """Call ``evaluate`` once per seed; return stacked scores ``(num_calls, num_instances)``."""
    samples = []
    for seed in seeds:
        torch.manual_seed(seed)
        samples.append(hpo_prob.evaluate(params).detach().clone())
    return torch.stack(samples, dim=0)


def _per_row_stats(samples: torch.Tensor):
    """Per-row mean / min / max / span / std / relative-std across the repeated calls."""
    mean = samples.mean(dim=0)
    lo, hi = samples.min(dim=0).values, samples.max(dim=0).values
    std = samples.std(dim=0)
    return mean, lo, hi, hi - lo, std, std / mean.abs()


def main() -> None:
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_printoptions(precision=4, sci_mode=False)
    t_start = time.perf_counter()

    print("=" * 78)
    print("EvoX HPO with repeated inner evaluation (num_repeats > 1)")
    print(f"device={torch.get_default_device()}  inner=PSO(pop={INNER_POP}) on Sphere(dim={DIM})  iterations={INNER_ITERATIONS}")
    print(f"outer instances={NUM_INSTANCES}  num_repeats={NUM_REPEATS}\n" + "=" * 78)

    # --- Stage 1: two wrappers around identical inner workflows, repeats 1 vs N ---
    torch.manual_seed(BUILD_SEED)
    hpo_single, _ = _build_hpo_problem(1)
    torch.manual_seed(BUILD_SEED)
    hpo_repeat, inner_repeat = _build_hpo_problem(NUM_REPEATS)
    mon = inner_repeat.monitor
    print("\n[1] Public HPO surface")
    print("    tunable keys (hpo_prob.get_params_keys()):", hpo_repeat.get_params_keys())
    print(f"    inner monitor: {type(mon).__name__}(num_repeats={mon.num_repeats})")
    print(f"    default fit_aggregation: {mon.fit_aggregation} -> the framework's MEAN aggregator")
    print("    (hpo_single uses num_repeats=1, so no aggregation is applied)")

    # --- Stage 2: identical, distinct hyper-parameter rows for both wrappers -----
    params = _make_params(hpo_repeat, seed=BUILD_SEED)
    print("\n[2] Hyper-parameter rows (one row = one outer candidate):")
    for key, value in params.items():
        print(f"    {key:<20} shape={tuple(value.shape)} values={value.detach().reshape(-1)}")

    # --- Stage 3: one `evaluate` call = ALL instances x ALL repeats -> ONE scalar per row
    torch.manual_seed(EVAL_SEED)
    agg_single = hpo_single.evaluate(params).detach()
    torch.manual_seed(EVAL_SEED)
    agg_repeat = hpo_repeat.evaluate(params).detach()
    assert agg_single.shape == (NUM_INSTANCES,) and agg_repeat.shape == (NUM_INSTANCES,), (
        "evaluate must return exactly one scalar per hyper-parameter row"
    )
    print("\n[3] Public evaluate(params) -> ONE scalar per hyper-parameter row")
    print(f"    num_repeats=1 : shape={tuple(agg_single.shape)}  dtype={agg_single.dtype}  values={agg_single}")
    print(f"    num_repeats={NUM_REPEATS} : shape={tuple(agg_repeat.shape)}  dtype={agg_repeat.dtype}  values={agg_repeat}")
    print("    -> Both are (num_instances,): with num_repeats>1 the N inner runs per row are")
    print("       collapsed by the monitor's aggregation into that single value.")

    # --- Stage 4: score the SAME rows several times (different seeds) and compare spread
    seeds = [EVAL_SEED + 10 * k for k in range(NUM_EVAL_CALLS)]
    stats_s = _per_row_stats(_sample(hpo_single, params, seeds))
    stats_r = _per_row_stats(_sample(hpo_repeat, params, seeds))
    print(f"\n[4] Score spread over {NUM_EVAL_CALLS} independent evaluate() calls (different seeds)")
    print("    per row: mean / min / max / span / std / std-relative-to-mean")
    for title, (mean, lo, hi, span, std, rel) in (
        ("num_repeats=1 (one inner run per row):", stats_s),
        (f"num_repeats={NUM_REPEATS} (MEAN over {NUM_REPEATS} inner runs per row):", stats_r),
    ):
        print(f"    {title}")
        for i in range(NUM_INSTANCES):
            print(f"      row {i}: mean={float(mean[i]):>8.4f} min={float(lo[i]):>8.4f} max={float(hi[i]):>8.4f}"
                  f" span={float(span[i]):>8.4f} std={float(std[i]):>7.4f} rel={float(rel[i]):>6.4f}")

    print(f"\n    mean over rows             |  num_repeats=1  |  num_repeats={NUM_REPEATS}")
    print(f"      absolute span            |  {float(stats_s[3].mean()):>12.4f}  |  {float(stats_r[3].mean()):>12.4f}")
    print(f"      absolute std             |  {float(stats_s[4].mean()):>12.4f}  |  {float(stats_r[4].mean()):>12.4f}")
    print(f"      relative std (std/|mean|) |  {float(stats_s[5].mean()):>12.4f}  |  {float(stats_r[5].mean()):>12.4f}")
    print(f"    rows whose span is smaller with repeats: {int((stats_r[3] < stats_s[3]).sum())}/{NUM_INSTANCES}")
    # Describe the direction from the numbers just measured (never hard-code a claim).
    abs_word = "smaller" if float(stats_r[4].mean()) < float(stats_s[4].mean()) else "larger"
    rel_word = "smaller" if float(stats_r[5].mean()) < float(stats_s[5].mean()) else "larger"
    print(
        f"    -> The num_repeats={NUM_REPEATS} wrapper runs the inner workflow {NUM_REPEATS} times per row; the monitor's\n"
        "       MEAN aggregator collapses those N runs into the single score of [3]. That score\n"
        "       sits on its own scale, so the two configurations do not estimate the same quantity\n"
        "       directly and their absolute spreads are not comparable. Measured here: absolute\n"
        f"       spread {abs_word} with repeats, scale-free relative spread {rel_word}. Averaging N\n"
        "       stochastic inner runs is the intended way to give the outer optimiser a smoother\n"
        "       target; these are the numbers this small configuration actually produced."
    )

    # --- Stage 5: a short outer optimisation loop (PSO over the tunable keys) -----
    class solution_transform(torch.nn.Module):
        def forward(self, x: torch.Tensor):
            return {"algorithm.w": x[:, 0], "algorithm.phi_p": x[:, 1], "algorithm.phi_g": x[:, 2]}

    torch.manual_seed(BUILD_SEED)
    outer_workflow = StdWorkflow(
        algorithm=PSO(pop_size=NUM_INSTANCES, lb=0 * torch.ones(3), ub=3 * torch.ones(3)),
        problem=hpo_repeat,  # the repeated wrapper: the outer loop sees aggregated scores
        monitor=EvalMonitor(),
        solution_transform=solution_transform(),
    )
    outer_monitor = outer_workflow.monitor

    print(f"\n[5] Outer PSO over (w, phi_p, phi_g), {OUTER_GENERATIONS} generations")
    torch.manual_seed(EVAL_SEED)
    outer_workflow.init_step()
    print(f"    generation 0 (init): aggregated best = {float(outer_monitor.get_best_fitness()):.4f}")
    for generation in range(1, OUTER_GENERATIONS + 1):
        outer_workflow.step()
        print(f"    generation {generation}: aggregated best = {float(outer_monitor.get_best_fitness()):.4f}")

    best_hyper = outer_monitor.get_best_solution()
    print(f"\n    Final best hyper-parameters (w, phi_p, phi_g): "
          f"w={float(best_hyper[0]):.4f}  phi_p={float(best_hyper[1]):.4f}  phi_g={float(best_hyper[2]):.4f}")
    print(f"\nDone in {time.perf_counter() - t_start:.2f}s.")


if __name__ == "__main__":
    main()
