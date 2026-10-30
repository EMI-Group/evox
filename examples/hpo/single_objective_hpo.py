"""EvoX (evox_etl) single-objective hyperparameter optimization (HPO) example.

An *inner* optimization run (PSO on the 10-D Rastrigin function) is wrapped by
:class:`evox_etl.problems.hpo_wrapper.HPOProblemWrapper`, whose search space is
a set of the inner config's fields — the PSO weights

    w       inertia weight          (default 0.6)
    phi_p   cognitive (personal)    (default 2.5)
    phi_g   social (global)         (default 0.8)

A host-side random search (:func:`random_search`) samples candidates in
[0, 4]^3, evaluates each one by actually RUNNING the inner workflow
(``HPOProblemWrapper.evaluate`` -> ``StdWorkflow.run`` -> best inner fitness),
and reports the best hyperparameters found next to the untuned defaults.

Unlike an in-graph HPO design (which would nest the inner workflow inside the
outer workflow's traced ``evaluate``), this wrapper is host-side Python: the
outer search is a plain loop over ``evaluate``, and only the inner run is
compiled. See ``src/evox_etl/problems/hpo_wrapper.py`` and
``src/evox_etl/DESIGN.md`` for why a nested compiled workflow is impossible.

Only ``numpy`` + ``evoxc`` are needed; the inner workflow runs on the numpy
backend (``HPOProblemConfig(backend="numpy")`` — set ``backend="iree"``/``"xla"``
to compile the inner runs, which needs the matching evoxc backend extra, e.g.
``pip install "evoxc[iree]"``).

Run (CPU only, a few seconds):
    python examples/hpo/single_objective_hpo.py
"""

import dataclasses
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
INNER_POP = 20  # population size of the inner PSO
INNER_GENS = 30  # inner generations per hyperparameter candidate
# One (low, high) pair per tuned hyperparameter, in `HPSlot` order.
SEARCH_BOUNDS = [(0.0, 1.0), (0.0, 4.0), (0.0, 4.0)]  # w, phi_p, phi_g
N_TRIALS = 18  # candidates sampled by the random search
SEED = 42


def main() -> None:
    print("=" * 70)
    print("EvoX HPO (single-objective): tune PSO (w, phi_p, phi_g) on 10-D Rastrigin")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Stage 1: describe the INNER run and which of its fields are tuned.
    # ------------------------------------------------------------------
    inner_algorithm = PSO(pop_size=INNER_POP, lb=(-5.12,) * DIM, ub=(5.12,) * DIM)
    inner_problem = Rastrigin()  # global optimum 0 at the origin
    hpo = HPOProblemWrapper(
        HPOProblemConfig(
            algorithm=inner_algorithm,
            problem=inner_problem,
            # One HPSlot per tuned field, addressed as "<config>.<field>":
            # config_kind is "algorithm" / "problem" / "monitor".
            hyperparameters=(
                HPSlot("algorithm", "w"),
                HPSlot("algorithm", "phi_p"),
                HPSlot("algorithm", "phi_g"),
            ),
            num_generations=INNER_GENS,
            # The inner monitor must be an EvalMonitorConfig (HPO reads the inner
            # run's best fitness from it) wrapped in the HPO policy config.
            monitor=HPOFitnessMonitorConfig(inner_monitor=EvalMonitorConfig()),
            num_repeats=1,  # one inner run per candidate (see repeated_hpo.py)
            seed=SEED,  # base seed of the inner runs
            backend="numpy",  # inner workflow backend ("iree"/"xla" also work)
        )
    )

    keys = hpo.get_params_keys()  # e.g. ["algorithm.w", "algorithm.phi_p", ...]
    defaults = hpo.get_init_params()  # current values, taken from the inner config
    print(f"\n[inner] PSO(pop_size={INNER_POP}) on {DIM}-D Rastrigin, {INNER_GENS} generations per candidate")
    print("\n[hpo] tunable hyperparameters:")
    for key, (low, high) in zip(keys, SEARCH_BOUNDS):
        print(f"        {key:<18} default = {defaults[key]:<5.2f} search range = [{low}, {high}]")

    # ------------------------------------------------------------------
    # Stage 2: baseline — evaluate the inner config UNCHANGED.
    # `evaluate(None)` (or `evaluate({})`) keeps every field at its default.
    # ------------------------------------------------------------------
    start = time.perf_counter()
    baseline = hpo.evaluate(None)
    print(f"\n[base] default hyperparameters: best inner fitness = {baseline:.4f}"
          f"   ({time.perf_counter() - start:.2f}s)")

    # ------------------------------------------------------------------
    # Stage 3: search — uniform random sampling of the hyperparameter box,
    # each sample evaluated by one full inner run.
    # ------------------------------------------------------------------
    start = time.perf_counter()
    best_values, best_fitness = random_search(hpo, SEARCH_BOUNDS, n_trials=N_TRIALS, seed=SEED)
    elapsed = time.perf_counter() - start

    print(f"\n[search] {N_TRIALS} random candidates in {elapsed:.2f}s "
          f"({elapsed / N_TRIALS * 1000:.0f} ms per full inner run)")
    print("\n" + "-" * 70)
    print("Result")
    print("-" * 70)
    for key, value in zip(keys, best_values):
        print(f"  best {key:<18} = {float(value):.4f}   (default {defaults[key]:.2f})")
    print(f"  best inner fitness        = {best_fitness:.4f}")
    print(f"  default hyperparameters   = {baseline:.4f}")
    improvement = 100.0 * (baseline - best_fitness) / baseline if baseline != 0 else 0.0
    print(f"  improvement over defaults = {improvement:.1f}%")

    # `evaluate` is deterministic for a given (candidate, seed): re-evaluating
    # the reported candidate reproduces the reported score exactly.
    assert abs(hpo.evaluate(best_values) - best_fitness) < 1e-9

    # ------------------------------------------------------------------
    # Stage 4: use the tuned hyperparameters — rebuild the inner algorithm
    # config with the tuned field values and run it as usual.
    # ------------------------------------------------------------------
    tuned_fields = {key.rsplit(".", 1)[-1]: float(value) for key, value in zip(keys, best_values)}
    tuned_algorithm = dataclasses.replace(inner_algorithm, **tuned_fields)
    tuned_workflow = StdWorkflow(
        algorithm=tuned_algorithm,
        problem=inner_problem,
        monitor=EvalMonitorConfig(),
        opt_direction="min",
        num_generations=INNER_GENS,
        backend="numpy",
    )
    tuned_workflow.run(INNER_GENS, seed=SEED)
    reproduced = tuned_workflow.monitor.get_best_fitness()
    print(f"\n[apply] a StdWorkflow built with the tuned config scores {reproduced:.4f}"
          f" (HPO reported {best_fitness:.4f})")
    assert np.isclose(reproduced, best_fitness, atol=1e-6)


if __name__ == "__main__":
    main()
