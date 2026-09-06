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

## Gate (counts verified: 363 = 97 + 82 + 131 + 53, zero failures/errors/skips)
```
/mnt/local-ssd/bchuang/evox/.venv/bin/python -m pytest unit_test/etl/algorithms -q  # 97, ~9.5 min (torch-parity runs dominate)
/mnt/local-ssd/bchuang/evox/.venv/bin/python -m pytest unit_test/etl/operators -q   # 82, ~20 s
/mnt/local-ssd/bchuang/evox/.venv/bin/python -m pytest unit_test/etl/problems -q    # 131, ~70 s
/mnt/local-ssd/bchuang/evox/.venv/bin/python -m pytest unit_test/etl/metrics -q     # 53, ~11 s
```
Every `parity/` dir (and the `mo/`/`so/` subdirs) carries an `__init__.py`, so
the combined run `pytest unit_test/etl -q` also collects all 363 tests with no
basename collisions and passes green (verified ~11 min).
Suite-separate runs remain useful for per-suite numbers and faster failure isolation.

## Coverage notes (durable audit findings)
- NO test under this tree uses any etl backend other than `"numpy"` on CPU: no
  `backend="iree"/"xla"`, no `Device(...)`, no cuda.
  Compiled-backend/GPU paths are exercised only by the `benchmarks/etl_vs_torch/`
  harness (torch-cuda/etl-iree-llvm-cpu/etl-iree-cuda/etl-xla-cuda), not by unit tests.
- No `pytest.mark.skip`/`xfail` markers anywhere: every test that exists runs;
  nothing is silently skipped. Known-not-ported code (e.g. `virtual_lora_es`,
  asebo's iree/xla `etl.svd` blocker) simply has no module/test to exercise it —
  see `src/evox_etl/algorithms/CONTEXT.md` + `es_variants/CONTEXT.md`.
- `evox_etl`'s OWN `StdWorkflow`/`EvalMonitor` (`src/evox_etl/core/workflow.py`,
  `workflows/`) have NO tests here: algorithm tests drive raw init/ask/tell via
  `helpers.run_generations`, and parity tests drive the TORCH StdWorkflow.
  etl-workflow validation is ad hoc (`src/evox_etl/workflows/CONTEXT.md`) plus
  the benchmark harness.
- Algorithms with etl-only smoke coverage and no torch parity: code/jade/ode/
  sade/shade, ars/asebo/des/esmc/guided_es/nes/noise_reuse_es/persistent_es/
  snes, rvea/rveaa/hype (parity exists for de, pso, cma_es, open_es, nsga2,
  nsga3, moead — see `algorithms/parity/CONTEXT.md`).

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
