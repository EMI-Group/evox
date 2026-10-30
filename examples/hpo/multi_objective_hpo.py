"""EvoX (evox_etl) multi-objective hyperparameter optimization (HPO) example.

The inner run is a real multi-objective optimizer — RVEA on the 2-objective
DTLZ2 benchmark — and its hyperparameters are tuned by a host-side random
search. Because a Pareto front is not a single number, the inner run's quality
is collapsed into one scalar by ``multi_obj_metric``: the Inverted Generational
Distance (IGD) between the inner run's approximate Pareto front and the true
DTLZ2 front.

The tuned fields are the RVEA rates that shape the reference-vector
adaptation:

    alpha   penalty rate applied to the angle        (default 2.0)
    fr      frequency of the reference-vector update (default 0.1)

Two pieces of the recipe are worth reading closely:

* ``evox_etl.metrics.igd`` is an ``evoxc`` *traceable* function, so it cannot
  be called on numpy arrays directly (``evoxc`` has no eager mode). It is
  wrapped host-side once with ``evoxc.build`` + ``evoxc.run`` — exactly what
  ``hpo_wrapper``'s docstring prescribes. The compiled metric is cached per
  input shape, because the number of Pareto-front points varies per candidate.
* the inner monitor needs ``full_fit_history=True`` (the default) so
  ``get_pf_fitness()`` can derive the front, and ``opt_direction`` must be a
  list with one direction per objective.

Only ``numpy`` + ``evoxc`` are needed; the inner workflow runs on the numpy
backend (switch ``HPOProblemConfig(backend=...)`` to ``"iree"``/``"xla"`` to
compile the inner runs — those need the matching evoxc backend extra, e.g.
``pip install "evoxc[iree]"``).

Run (CPU only, a few seconds):
    python examples/hpo/multi_objective_hpo.py
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

import evoxc
from evoxc.core import TensorSpec
from evox_etl import EvalMonitorConfig, StdWorkflow
from evox_etl.algorithms import RVEA
from evox_etl.metrics import igd
from evox_etl.problems.hpo_wrapper import (
    HPOFitnessMonitorConfig,
    HPOProblemConfig,
    HPOProblemWrapper,
    HPSlot,
    random_search,
)
from evox_etl.problems.numerical import DTLZ2
from evox_etl.problems.numerical.dtlz import pf as dtlz_pf

# ---------------------------------------------------------------------------
# Configuration (kept small so the whole script runs in a few seconds).
# ---------------------------------------------------------------------------
DIM = 5  # decision dimension of DTLZ2
N_OBJS = 2
POP_SIZE = 20  # RVEA population size
INNER_GENS = 30  # inner generations per hyperparameter candidate
SEARCH_BOUNDS = [(0.5, 4.0), (0.05, 0.4)]  # alpha, fr
N_TRIALS = 8  # candidates sampled by the random search
SEED = 42


def true_pareto_front(problem: DTLZ2) -> np.ndarray:
    """The reference DTLZ2 front as a numpy array ``(ref_num * m, m)``.

    The problem's ``pf`` function is an ``evoxc`` traceable function, so it is
    built once and run once, host-side.
    """
    return evoxc.run(evoxc.build(dtlz_pf, problem), problem).numpy()


class HostIGD:
    """Host-side IGD to a fixed reference front (``evox_etl.metrics.igd``).

    ``evoxc`` ops only run inside a trace, so the metric is compiled once per
    ``(n_pareto, n_objs)`` shape and then called on numpy arrays. The front size
    changes from candidate to candidate, hence the per-shape cache.
    """

    def __init__(self, front: np.ndarray) -> None:
        """Store the reference front and its concrete evoxc tensor."""
        self.front = np.asarray(front, dtype=np.float32)
        self._front_tensor = evoxc.core.tensor(self.front)
        self._exes: dict[tuple[int, int], object] = {}

    def _exe(self, shape: tuple[int, int]):
        """Compiled ``igd(front, objs)`` executable for one ``objs`` shape."""
        exe = self._exes.get(shape)
        if exe is None:
            exe = evoxc.build(
                igd,
                TensorSpec(self.front.shape, np.float32),
                TensorSpec(shape, np.float32),
                backend="numpy",
            )
            self._exes[shape] = exe
        return exe

    def __call__(self, pareto_fitness: np.ndarray) -> np.ndarray:
        """IGD of an ``(n_pareto, n_objs)`` fitness array (0-d numpy array)."""
        objs = np.asarray(pareto_fitness, dtype=np.float32)
        out = evoxc.run(self._exe(objs.shape), self._front_tensor, evoxc.core.tensor(objs))
        return np.asarray(out.numpy())


def main() -> None:
    print("=" * 70)
    print("EvoX HPO (multi-objective): tune RVEA (alpha, fr) on 2-objective DTLZ2")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Stage 1: the INNER run plus the scalar quality score.
    # ------------------------------------------------------------------
    problem = DTLZ2(d=DIM, m=N_OBJS, ref_num=100)  # 200 reference front points
    front = true_pareto_front(problem)
    host_igd = HostIGD(front)
    inner_algorithm = RVEA(
        pop_size=POP_SIZE,
        n_objs=N_OBJS,
        lb=(0.0,) * DIM,
        ub=(1.0,) * DIM,
        alpha=2.0,  # RVEA default
        fr=0.1,  # RVEA default
        max_gen=INNER_GENS,  # RVEA adapts the reference vectors relative to max_gen
    )
    hpo = HPOProblemWrapper(
        HPOProblemConfig(
            algorithm=inner_algorithm,
            problem=problem,
            hyperparameters=(HPSlot("algorithm", "alpha"), HPSlot("algorithm", "fr")),
            num_generations=INNER_GENS,
            monitor=HPOFitnessMonitorConfig(
                # `get_pf_fitness()` needs the per-generation fitness history.
                inner_monitor=EvalMonitorConfig(full_fit_history=True),
                # Collapse the inner Pareto front into one scalar: IGD (smaller is better).
                multi_obj_metric=host_igd,
            ),
            # A list with one direction per objective marks the inner run as multi-objective.
            opt_direction=["min"] * N_OBJS,
            seed=SEED,
            backend="numpy",
        )
    )

    keys = hpo.get_params_keys()
    defaults = hpo.get_init_params()
    print(f"\n[inner] RVEA(pop_size={POP_SIZE}) on {N_OBJS}-objective DTLZ2 (d={DIM}), "
          f"{INNER_GENS} generations per candidate")
    print(f"[inner] quality score: IGD to the true front ({len(front)} reference points), lower is better")
    print("\n[hpo] tunable hyperparameters:")
    for key, (low, high) in zip(keys, SEARCH_BOUNDS):
        print(f"        {key:<18} default = {defaults[key]:<5.2f} search range = [{low}, {high}]")

    # ------------------------------------------------------------------
    # Stage 2: baseline — the inner run with its UNCHANGED hyperparameters.
    # ------------------------------------------------------------------
    start = time.perf_counter()
    baseline = hpo.evaluate(None)
    print(f"\n[base] default hyperparameters: IGD = {baseline:.5f}"
          f"   ({time.perf_counter() - start:.2f}s)")

    # ------------------------------------------------------------------
    # Stage 3: search — uniform random sampling, one full inner run per sample.
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
    print(f"  best IGD                   = {best_fitness:.5f}")
    print(f"  default hyperparameters    = {baseline:.5f}")
    improvement = 100.0 * (baseline - best_fitness) / baseline if baseline != 0 else 0.0
    print(f"  IGD reduction over defaults = {improvement:.1f}%")

    # ------------------------------------------------------------------
    # Stage 4: use the tuned hyperparameters — rebuild the inner config and
    # measure its Pareto front directly.
    # ------------------------------------------------------------------
    tuned_fields = {key.rsplit(".", 1)[-1]: float(value) for key, value in zip(keys, best_values)}
    tuned_algorithm = dataclasses.replace(inner_algorithm, **tuned_fields)
    tuned_workflow = StdWorkflow(
        algorithm=tuned_algorithm,
        problem=problem,
        monitor=EvalMonitorConfig(full_fit_history=True),
        opt_direction=["min"] * N_OBJS,
        num_generations=INNER_GENS,
        backend="numpy",
    )
    tuned_workflow.run(INNER_GENS, seed=SEED)
    tuned_pareto = tuned_workflow.monitor.get_pf_fitness()
    reproduced = float(host_igd(tuned_pareto))
    print(f"\n[apply] a StdWorkflow built with the tuned config reaches {len(tuned_pareto)} "
          f"Pareto points, IGD = {reproduced:.5f} (HPO reported {best_fitness:.5f})")
    assert np.isclose(reproduced, best_fitness, atol=1e-5)
    # The DTLZ2 front is the unit sphere, so an optimal front has |f| == 1.
    print(f"        max |f| deviation from the true front: "
          f"{np.abs(np.linalg.norm(tuned_pareto, axis=1) - 1.0).max():.5f}")


if __name__ == "__main__":
    main()
