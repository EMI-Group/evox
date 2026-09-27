# evox_etl problems — test suite

Pytest suite for the functional `evox_etl` problems, running everything through
`etl.build` + `etl.run` on the numpy backend. The pure-etl tests (no torch) live
at the top level:

- `test_basic.py`, `test_dtlz.py`, `test_cec2022.py` — the numerical problems
  (`basic`, `dtlz`, `cec2022`).
- `test_virtual_problem.py` — the neuroevolution virtual Gaussian-noise
  (training-free) problem.
- `test_hpo_wrapper.py` — the host-side HPO wrapper (inner `StdWorkflow` +
  `HPOProblemWrapper` / `HPOFitnessMonitor` / `random_search`).

`parity/` compares `evox_etl` against the pip-installed torch `evox` reference
and is the only place allowed to import torch. The `conftest.py` sys.path shim
locates the repository root from `__file__`, so no path adjustments are needed.

Run (from the repository root):

```
/mnt/local-ssd/bchuang/evox/.venv/bin/python -m pytest unit_test/etl/problems -q
```
