"""Release gate: importing and running the EvoX port must NOT pull in torch.

The torch implementation of EvoX 1.x is gone from this repository, yet the
gate is still worth keeping: a stray optional dependency or a careless helper
could drag the heavy ``torch`` package (and its GPU stack) into a release that
only needs ``evoxc`` + ``numpy``. An in-process ``"torch" not in sys.modules``
check would be meaningless — this very pytest session may have loaded it — so
the gate runs a FRESH interpreter (``sys.executable -c ...``) with
``PYTHONPATH`` pointing at the repository's ``src`` directory, imports
``evox``, runs a small PSO-on-Sphere optimization through
``evox.workflows.StdWorkflow`` and only then asserts that no torch module
entered ``sys.modules``.

No network, numpy backend only.
"""
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"

_GATE_SCRIPT = r"""
import sys

import numpy as np

import evox
from evox.algorithms import make_pso
from evox.problems.numerical.basic import Sphere
from evox.workflows import EvalMonitorConfig, StdWorkflow

assert "torch" not in sys.modules, sorted(m for m in sys.modules if "torch" in m)

wf = StdWorkflow(
    make_pso(pop_size=10, lb=[-5.0] * 4, ub=[5.0] * 4),
    Sphere(),
    monitor=EvalMonitorConfig(),
    opt_direction="min",
    num_generations=5,
)
wf.run(seed=0)
best = wf.monitor.get_best_fitness()
assert np.isfinite(best), best

leaked = sorted(m for m in sys.modules if m == "torch" or m.startswith("torch."))
assert not leaked, "%d torch modules imported, e.g. %r" % (len(leaked), leaked[:5])
print("OK best=%r" % (best,))
"""


def _run_gate():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(_SRC), str(_REPO_ROOT)])
    return subprocess.run(
        [sys.executable, "-c", _GATE_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(_REPO_ROOT),
    )


def test_import_and_optimization_do_not_import_torch():
    proc = _run_gate()
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "OK" in proc.stdout, proc.stdout
