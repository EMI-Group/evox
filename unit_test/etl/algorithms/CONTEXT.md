# unit_test/etl/algorithms — algorithm smoke/parity tests (etl-only)

## Intent
Shared scaffolding + tests for the functional evox_etl algorithm port
(`src/evox_etl/algorithms/`).  No torch imports here — parity tests that
import torch live in `unit_test/etl/parity/`.

- `conftest.py` — sys.path shim making the repo root + `src/` importable.
- `helpers.py` — toy problems (Sphere/Rosenbrock/Ackley/DTLZ1) + generic
  `run_generations` init/ask/tell driver on the etl `"numpy"` backend.
- `test_helpers.py` — tests for the scaffolding itself (pytest).

## Constraints
- ETL has no eager mode: all ops inside functions traced via
  `etl.build`/`etl.run` (backend `"numpy"`).
- `etl.run` requires ALL signature args, static dataclasses included.
- Scalar graph inputs are 0-d numpy arrays; reductions use `axes=`.
