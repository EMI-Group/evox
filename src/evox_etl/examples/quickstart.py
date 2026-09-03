"""EvoX-ETL quickstart: functional PSO on the Sphere function.

End-to-end demo of the functional rewrite of EvoX on the ETL tensor library:
real PSO (``evox_etl.algorithms``) + real Sphere (``evox_etl.problems``) +
EvalMonitor + StdWorkflow, 50 generations on the numpy backend.

Intended to live at the repo root as ``examples/quickstart.py``; the sys.path
bootstrap below locates the repo root by walking up to the directory that
contains ``src/evox_etl/__init__.py``, so the script works both from
``src/evox_etl/examples/`` (this location) and from a repo-root ``examples/``.

Run (no installation needed):
    /mnt/local-ssd/bchuang/evox/.venv/bin/python examples/quickstart.py
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

from evox_etl import StdWorkflow, EvalMonitorConfig
from evox_etl.algorithms import PSO
from evox_etl.problems.numerical import Sphere


def main() -> None:
    dim = 10
    lb = np.full(dim, -100.0)
    ub = np.full(dim, 100.0)

    algorithm = PSO(pop_size=100, lb=lb, ub=ub)  # w=0.6, phi_p=2.5, phi_g=0.8
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

    print(f"EvoX-ETL quickstart: PSO (pop_size=100, dim={dim}) on Sphere")
    print("Optimizing for 50 generations (minimization, numpy backend)...\n")

    state = workflow.init(seed=42)
    for generation in range(1, 51):
        state = workflow.step(state)
        if generation % 10 == 0:
            best_fitness = workflow.monitor.get_best_fitness()
            best_fitness = float(np.asarray(best_fitness).reshape(-1)[0])
            print(f"  generation {generation:>2}: best fitness = {best_fitness:.6f}")

    best_fitness = float(np.asarray(workflow.monitor.get_best_fitness()).reshape(-1)[0])
    best_solution = np.asarray(workflow.monitor.get_best_solution())
    print(f"\nFinal best fitness: {best_fitness:.6f}")
    print(
        f"Best solution L2 norm: {np.linalg.norm(best_solution):.6f} "
        f"(true optimum is the origin)"
    )


if __name__ == "__main__":
    main()
