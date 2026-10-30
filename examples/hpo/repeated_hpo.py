"""EvoX (evox_etl) HPO example: repeated evaluation and fitness aggregation.

Real optimization runs are noisy: the same hyperparameters give a different
final fitness on a different random seed. ``HPOProblemWrapper`` therefore
supports ``num_repeats > 1`` — every candidate is evaluated by ``num_repeats``
independent inner runs (repeat ``r`` uses seed ``config.seed + r``) and the
per-repeat scores are combined by the monitor's ``fit_aggregation`` policy
(``"mean"``, ``"min"``, ``"max"`` or any ``numpy.ndarray -> float`` callable).

This script tunes the three PSO weights (``w``, ``phi_p``, ``phi_g``) of an
inner PSO on the 10-D Rastrigin function (a rugged landscape, so the seed
matters) and shows

1. how much one candidate's score really moves across seeds (the spread),
2. what each aggregation policy reports for that same candidate,
3. why searching with the seed-averaged score picks a more reliable
   hyperparameter set than searching with a single lucky run.

Only ``numpy`` + ``evoxc`` are needed; the inner runs use the numpy backend.

Run (CPU only, a few seconds):
    python examples/hpo/repeated_hpo.py
"""

import pathlib
import sys
import time

# Make `evox_etl` importable without an editable install.
_here = pathlib.Path(__file__).resolve()
for _p in _here.parents:
    if (_p / "src" / "evox_etl" / "__init__.py").is_file():
        _ROOT = _p
        break
else:  # pragma: no cover
    raise RuntimeError("could not locate the repo root (src/evox_etl/__init__.py)")
for _path in (str(_ROOT), str(_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np

from evox_etl import EvalMonitorConfig, StdWorkflow
from evox_etl.algorithms import PSO
from evox_etl.problems.hpo_wrapper import (
    HPOFitnessMonitorConfig,
    HPOProblemConfig,
    HPOProblemWrapper,
    HPSlot,
    random_search,
)
from evox_etl.problems.numerical import Rastrigin

# ---------------------------------------------------------------------------
# Configuration (kept small so the whole script runs in a few seconds).
# ---------------------------------------------------------------------------
DIM = 10  # dimension of the inner Rastrigin problem
INNER_POP = 15
INNER_GENS = 25  # inner generations per repeat
SEARCH_BOUNDS = [(0.0, 1.0), (0.0, 4.0), (0.0, 4.0)]  # w, phi_p, phi_g
N_TRIALS = 12  # candidates sampled by the random search
NUM_REPEATS = 3  # independent inner runs per candidate
SEED = 42


def build_wrapper(num_repeats: int, fit_aggregation: str = "mean") -> HPOProblemWrapper:
    """An HPO wrapper over the inner PSO-on-Rastrigin run with a repeat policy."""
    return HPOProblemWrapper(
        HPOProblemConfig(
            algorithm=PSO(pop_size=INNER_POP, lb=(-5.12,) * DIM, ub=(5.12,) * DIM),
            problem=Rastrigin(),
            hyperparameters=(
                HPSlot("algorithm", "w"),
                HPSlot("algorithm", "phi_p"),
                HPSlot("algorithm", "phi_g"),
            ),
            num_generations=INNER_GENS,
            monitor=HPOFitnessMonitorConfig(
                inner_monitor=EvalMonitorConfig(),
                # How the per-repeat best-fitness values are combined.
                fit_aggregation=fit_aggregation,
            ),
            num_repeats=num_repeats,
            seed=SEED,  # repeat r uses seed + r
            backend="numpy",
        )
    )


def per_seed_scores(candidate, keys, seeds) -> list[float]:
    """Best fitness of the inner run for `candidate` on each seed (fresh run per seed)."""
    fields = {key.rsplit(".", 1)[-1]: float(value) for key, value in zip(keys, candidate)}
    algorithm = PSO(pop_size=INNER_POP, lb=(-5.12,) * DIM, ub=(5.12,) * DIM, **fields)
    scores = []
    for seed in seeds:
        # A fresh workflow per seed: monitor history must not leak between runs.
        workflow = StdWorkflow(
            algorithm=algorithm,
            problem=Rastrigin(),
            monitor=EvalMonitorConfig(),
            opt_direction="min",
            num_generations=INNER_GENS,
            backend="numpy",
        )
        workflow.run(INNER_GENS, seed=seed)
        scores.append(workflow.monitor.get_best_fitness())
    return scores


CANDIDATE_KEYS: list[str] = []  # filled in main(); used by per_seed_scores


def main() -> None:
    print("=" * 74)
    print("EvoX HPO (repeated evaluation): tune PSO (w, phi_p, phi_g) on 10-D Rastrigin")
    print("=" * 74)

    keys = build_wrapper(1).get_params_keys()
    defaults = build_wrapper(1).get_init_params()
    candidate = [0.7, 2.0, 1.0]  # a fixed, reasonable-looking (w, phi_p, phi_g)

    # ------------------------------------------------------------------
    # Stage 1: how noisy is one candidate? Run the SAME hyperparameters on
    # three seeds (repeat r of the wrapper uses seed + r).
    # ------------------------------------------------------------------
    seeds = [SEED + repeat for repeat in range(NUM_REPEATS)]
    scores = per_seed_scores(candidate, keys, seeds)
    print(f"\n[spread] candidate {dict(zip(keys, candidate))} on {NUM_REPEATS} seeds:")
    for seed, score in zip(seeds, scores):
        print(f"           seed {seed}: best inner fitness = {score:.4f}")
    print(f"           spread: min = {min(scores):.4f}, mean = {np.mean(scores):.4f}, "
          f"max = {max(scores):.4f} (max/min = {max(scores) / min(scores):.1f}x)")

    # ------------------------------------------------------------------
    # Stage 2: the aggregation policies, on that same candidate.
    # All three wrappers run the identical inner runs; only the combination
    # of the per-repeat scores differs.
    # ------------------------------------------------------------------
    print("\n[aggregation] the same candidate, scored three ways:")
    for policy in ("mean", "min", "max"):
        wrapper = build_wrapper(NUM_REPEATS, fit_aggregation=policy)
        start = time.perf_counter()
        score = wrapper.evaluate(candidate)
        print(f"           num_repeats={NUM_REPEATS}, fit_aggregation={policy:<5} -> {score:.4f}"
              f"   ({time.perf_counter() - start:.2f}s)")

    single_run = build_wrapper(1, fit_aggregation="mean")
    start = time.perf_counter()
    single_score = single_run.evaluate(candidate)
    print(f"           num_repeats=1                        -> {single_score:.4f}"
          f"   ({time.perf_counter() - start:.2f}s)")
    print("           ('min' is the most optimistic and 'max' the most pessimistic policy;")
    print("            'mean' is the seed-robust score the search should optimize.)")

    # ------------------------------------------------------------------
    # Stage 3: search with the seed-averaged score.
    # ------------------------------------------------------------------
    robust = build_wrapper(NUM_REPEATS, fit_aggregation="mean")
    start = time.perf_counter()
    baseline = robust.evaluate(None)  # the inner config's default hyperparameters
    best_values, best_fitness = random_search(robust, SEARCH_BOUNDS, n_trials=N_TRIALS, seed=SEED)
    elapsed = time.perf_counter() - start

    print(f"\n[search] {N_TRIALS} candidates x {NUM_REPEATS} repeats in {elapsed:.2f}s")
    print("\n" + "-" * 74)
    print("Result (num_repeats=%d, fit_aggregation='mean')" % NUM_REPEATS)
    print("-" * 74)
    for key, value in zip(CANDIDATE_KEYS, best_values):
        print(f"  best {key:<18} = {float(value):.4f}   (default {defaults[key]:.2f})")
    print(f"  best seed-averaged fitness = {best_fitness:.4f}")
    print(f"  default hyperparameters    = {baseline:.4f}")
    improvement = 100.0 * (baseline - best_fitness) / baseline if baseline != 0 else 0.0
    print(f"  improvement over defaults  = {improvement:.1f}% (seed-averaged)")

    # The single-run score of the same tuned candidate is one sample of the
    # distribution the mean averages over, so it may be better OR worse.
    tuned_single = single_run.evaluate(best_values)
    print(f"  one single-seed run of the tuned candidate = {tuned_single:.4f}")
    print("  (a single run can look better or worse than the mean — which is exactly")
    print("   why hyperparameters chosen on one seed often fail to transfer.)")


if __name__ == "__main__":
    main()
