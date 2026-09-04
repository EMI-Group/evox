# unit_test/etl — tests for the functional evox_etl package

## Intent
pytest suite mirroring `src/evox_etl/` (functional EvoX on ETL). Four sub-suites:
1. `algorithms/` — algorithm smoke/parity tests + converted operator-shim tests
   (canonical `evox_etl.operators.*` imports; etl-only, no torch).
2. `operators/` — operator property tests (random ops, no torch) + `parity/`
   (exact parity vs torch `evox` within 1e-6; torch imports ONLY there).
3. `problems/` — numerical problem tests + `parity/` (relocated from
   `src/evox_etl/problems/tests/`; source copy pending deletion by another agent).
4. `metrics/` — metric tests + `parity/` (relocated from `src/evox_etl/metrics/tests/`).

## Gate (run each suite SEPARATELY — basename collisions across `parity/` dirs)
```
/mnt/local-ssd/bchuang/evox/.venv/bin/python -m pytest unit_test/etl/algorithms -q  # 97
/mnt/local-ssd/bchuang/evox/.venv/bin/python -m pytest unit_test/etl/operators -q   # 82
/mnt/local-ssd/bchuang/evox/.venv/bin/python -m pytest unit_test/etl/problems -q    # 131
/mnt/local-ssd/bchuang/evox/.venv/bin/python -m pytest unit_test/etl/metrics -q     # 53
```

## Packaging notes
- Every suite dir and its `parity/` subdir has an empty `__init__.py` so pytest
  treats them as distinct packages (`problems.parity.test_parity` vs
  `metrics.parity.test_parity` etc.) — keep them if adding new parity dirs.
- Repo-root `conftest.py` shims sys.path (repo root + `src/`); each `parity/` and
  the problems/metrics suites also carry idempotent relocate-ready conftest shims.
- Test pattern: `etl.build(fn, *specs, backend="numpy")` + `etl.run(exe, *args)`;
  scalar tensor inputs are 0-d ndarrays (numpy scalars rejected at run boundary);
  statics re-passed at run in signature order; results need `.numpy()`.

## Known operator deviations (reported to root agent, tests adapted)
- Canonical `apd_fn` gathers with `relu(x)` where torch uses `norm_obj[x]`
  (negative-index wrap) — latent in `ref_vec_guided`; details in
  `operators/CONTEXT.md` and `algorithms/CONTEXT.md`.
- Canonical `DE_differential_sum` second output dtype varies (int32 replace=True,
  int64 replace=False) — torch-faithful promotion.

See `../../src/evox_etl/DESIGN.md` §6 for the testing strategy.
