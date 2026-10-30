"""EvoX quickstart — the functional ``evox_etl`` API on the ``evoxc`` compiler.

Two small end-to-end demos, both on the default numpy backend:

1. **Single-objective** — real PSO (``evox_etl.algorithms``) on the 10-D Sphere
   function (``evox_etl.problems``), driven with ``StdWorkflow.run``.
2. **Multi-objective** — NSGA2 on 2-objective DTLZ2, driven generation by
   generation with ``StdWorkflow.init``/``init_step``/``step``, reporting the
   approximate Pareto front and plotting it with ``EvalMonitor.plot`` (Plotly is
   optional: the plot is skipped when it is not installed).

Only ``numpy`` + ``evoxc`` are needed. Compiled backends are a one-word change:
``StdWorkflow(..., backend="iree")`` (or ``"xla"``) instead of ``"numpy"`` — the
algorithm/problem configs stay exactly the same (those backends need their evoxc
extras, e.g. ``pip install "evoxc[iree]"``).

Run (no installation needed — the bootstrap below locates the repo root):
    python examples/quickstart.py
"""

import pathlib
import sys

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
from evox_etl import EvalMonitorConfig, StdWorkflow
from evox_etl.algorithms import NSGA2, PSO
from evox_etl.problems.numerical import DTLZ2, Sphere
from evox_etl.problems.numerical.dtlz import pf as dtlz_pf


def single_objective_demo() -> None:
    """PSO on the 10-D Sphere function, driven with ``StdWorkflow.run``."""
    dim = 10
    # PSO config: bounds are plain float tuples (dim entries), the weights use
    # the PSO defaults (w=0.6, phi_p=2.5, phi_g=0.8).
    algorithm = PSO(pop_size=100, lb=(-100.0,) * dim, ub=(100.0,) * dim)
    problem = Sphere()  # minimum at x = [0, ..., 0], fitness 0
    monitor = EvalMonitorConfig()  # full fitness history, top-1 elite

    workflow = StdWorkflow(
        algorithm=algorithm,
        problem=problem,
        monitor=monitor,
        opt_direction="min",
        num_generations=50,
        backend="numpy",
    )

    print("=" * 70)
    print(f"[1/2] Single-objective: PSO (pop_size=100) on {dim}-D Sphere")
    print("=" * 70)

    # `run` = init + 50 generations; the monitor records every generation.
    workflow.run(50, seed=42)

    fitness_history = workflow.monitor.get_fitness_history()  # one array per generation
    for generation in range(9, 50, 10):
        print(f"  generation {generation + 1:>2}: best fitness = {np.min(fitness_history[generation]):.6f}")

    best_fitness = workflow.monitor.get_best_fitness()  # Python float
    best_solution = workflow.monitor.get_best_solution()  # numpy array
    print(f"\n  final best fitness:   {best_fitness:.6f}   (optimum 0)")
    print(f"  |best solution|:      {np.linalg.norm(best_solution):.6f}   (optimum 0)")
    print()


def multi_objective_demo() -> None:
    """NSGA2 on 2-objective DTLZ2, driven step by step, plus an optional plot."""
    dim = 5
    problem = DTLZ2(d=dim, m=2, ref_num=100)  # true front = unit sphere, ref_num points per obj
    algorithm = NSGA2(pop_size=40, n_objs=2, lb=(0.0,) * dim, ub=(1.0,) * dim)
    monitor = EvalMonitorConfig()

    workflow = StdWorkflow(
        algorithm=algorithm,
        problem=problem,
        monitor=monitor,
        opt_direction=["min", "min"],  # one direction per objective -> multi-objective
        num_generations=50,
        backend="numpy",
    )

    print("=" * 70)
    print(f"[2/2] Multi-objective: NSGA2 (pop_size=40) on {dim}-D DTLZ2 (2 objectives)")
    print("=" * 70)

    # The same loop `run` performs, spelled out: init -> init_step -> step ...
    # (DTLZ1 is the harder sibling of DTLZ2 and needs far more generations.)
    state = workflow.init(seed=42)
    state = workflow.init_step(state)
    for _ in range(49):
        state = workflow.step(state)
    workflow.flush()  # only needed when driving init/step by hand: append the last history entry

    pareto_fitness = workflow.monitor.get_pf_fitness()  # (n_pareto, n_objs), un-negated
    # The DTLZ2 Pareto front is the unit sphere, so |f| == 1 for every optimal point.
    front_error = np.abs(np.linalg.norm(pareto_fitness, axis=1) - 1.0)
    print(f"  approximate Pareto front: {len(pareto_fitness)} points")
    print(f"  max |f| deviation from the true front: {front_error.max():.6f}")
    print(f"  objective ranges: f1 in [{pareto_fitness[:, 0].min():.4f}, {pareto_fitness[:, 0].max():.4f}]"
          f", f2 in [{pareto_fitness[:, 1].min():.4f}, {pareto_fitness[:, 1].max():.4f}]")

    # Plotly is an optional dependency: `plot` warns and returns None without it.
    true_front = evoxc.run(evoxc.build(dtlz_pf, problem), problem).numpy()
    figure = workflow.monitor.plot(problem_pf=true_front)
    if figure is None:
        print("\n  (plot skipped: install plotly with `pip install plotly` to enable EvalMonitor.plot)")
    else:
        generations = len(figure.frames) if figure.frames else 1
        print(f"\n  plotly figure ready: {len(figure.data)} traces over {generations} generations")
        print("  (interactive window: `figure.show()`; standalone html: `figure.write_html('pareto.html')`)")


def main() -> None:
    single_objective_demo()
    multi_objective_demo()


if __name__ == "__main__":
    main()
