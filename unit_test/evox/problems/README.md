# evox problems — test suite

Pytest suite for the functional `evox` problems, running everything through
`evoxc.build` + `evoxc.run` on the numpy backend (no torch anywhere):
- `test_basic.py`, `test_dtlz.py`, `test_cec2022.py` — the numerical problems
  (`basic`, `dtlz`, `cec2022`).
- `test_virtual_problem.py` — the neuroevolution virtual Gaussian-noise
  (training-free) problem.
- `test_hpo_wrapper.py` — the host-side HPO wrapper (inner `StdWorkflow` +
  `HPOProblemWrapper` / `HPOFitnessMonitor` / `random_search`).

The `conftest.py` sys.path shim locates the repository root from `__file__`, so
no path adjustments are needed.

Run (from the repository root):
```
PYTHONPATH=src <venv>/bin/python -m pytest unit_test/evox/problems -q
```
