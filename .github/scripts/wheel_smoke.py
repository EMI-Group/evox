"""Wheel-only smoke test for the evox 2.0 pre-release wheel.

CI runs this from a directory *outside* the repository checkout
(``$RUNNER_TEMP``) so that ``import evox_etl`` resolves to the installed wheel
instead of the source tree: a wheel that lost its package data (the
``cec2022_input_data/*.txt`` matrices read by ``CEC2022``) fails here and
nowhere else, because the test suite imports the sources.

What it proves, in order:
1. the wheel imports from ``site-packages`` and carries its submodules;
2. a real 5-generation PSO run on 10-D ``Sphere`` through ``StdWorkflow``
   (numpy backend) produces a finite best fitness;
3. ``CEC2022(problem_number=1, dimension=10)`` evaluates both through
   ``StdWorkflow`` and through the raw ``evoxc.build``/``evoxc.run`` step
   protocol, i.e. the shipped data files are found at runtime.

Run locally with::

    python -m build --outdir dist
    pip install "dist/evox-*.whl[test]"
    cd /tmp && python /path/to/checkout/.github/scripts/wheel_smoke.py
"""

import pathlib

import evoxc
import numpy as np

import evox_etl
from evox_etl import EvalMonitorConfig, StdWorkflow
from evox_etl.algorithms import PSO
from evox_etl.problems.numerical import CEC2022, Sphere
from evox_etl.problems.numerical.cec2022 import evaluate as cec2022_evaluate
from evox_etl.problems.numerical.state import ProblemState


def check_imported_from_wheel() -> None:
    """Fail if ``evox_etl`` resolved to a source checkout instead of the wheel."""
    root = pathlib.Path(evox_etl.__file__).resolve()
    print(f"evox_etl imported from: {root}")
    assert root.parent.parent.name == "site-packages", f"expected an installed wheel, got {root}"


def check_pso_on_sphere() -> None:
    """5 generations of PSO on 10-D Sphere through ``StdWorkflow.run``."""
    dim = 10
    workflow = StdWorkflow(
        PSO(pop_size=50, lb=(-100.0,) * dim, ub=(100.0,) * dim),
        Sphere(),
        monitor=EvalMonitorConfig(),
        opt_direction="min",
        num_generations=5,
        backend="numpy",
    )
    workflow.run(5, seed=42)
    best = workflow.monitor.get_best_fitness()
    print(f"PSO on 10-D Sphere, 5 generations: best fitness = {best:.6f}")
    assert np.isfinite(best)


def check_cec2022() -> None:
    """CEC2022 f1 at D=10 — the package-data check (shift/rotation from .txt files)."""
    dim = 10
    cec = CEC2022(problem_number=1, dimension=dim)

    # (a) through the workflow: PSO evaluates the whole population once.
    workflow = StdWorkflow(
        PSO(pop_size=20, lb=(-100.0,) * dim, ub=(100.0,) * dim),
        cec,
        monitor=EvalMonitorConfig(),
        opt_direction="min",
        num_generations=1,
        backend="numpy",
    )
    workflow.run(1, seed=0)
    best = workflow.monitor.get_best_fitness()
    print(f"CEC2022 f1 (D=10) via StdWorkflow: best fitness = {best:.6f}")
    assert np.isfinite(best)

    # (b) through the raw step protocol on a fixed population.
    pop = np.zeros((4, dim), dtype=np.float32)
    spec = evoxc.core.TensorSpec(pop.shape, np.dtype("float32"))
    exe = evoxc.build(cec2022_evaluate, cec, ProblemState(), spec)
    fitness, _ = evoxc.run(exe, cec, ProblemState(), pop)
    values = np.asarray(fitness.numpy())
    print(f"CEC2022 f1 (D=10) via evoxc.run:   fitness = {values}")
    assert np.all(np.isfinite(values))


def main() -> None:
    check_imported_from_wheel()
    check_pso_on_sphere()
    check_cec2022()
    print("wheel smoke test OK")


if __name__ == "__main__":
    main()
