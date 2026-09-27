# unit_test/etl — tests for the functional evox_etl package

## Intent
pytest suite mirroring `src/evox_etl/` (functional EvoX on ETL). Seven sub-suites:
1. `algorithms/` — algorithm smoke/parity tests + converted operator-shim tests
   (canonical `evox_etl.operators.*` imports; etl-only, no torch except `parity/`).
   Includes the virtual-ES / VirtualLoRA-ES smoke tests and the virtual
   end-to-end convergence tests under `so/es_variants/`; `parity/` also holds the
   ES-variant ASEBO/ARS/CMA-ES STATE-level parity file (`test_parity.py`).
2. `operators/` — operator property tests (random ops, no torch) + `parity/`
   (exact parity vs torch `evox` within 1e-6; torch imports ONLY there).
3. `problems/` — numerical problem tests + the neuroevolution virtual-problem
   suite (`test_virtual_problem.py`) + the host-side HPO-wrapper suite
   (`test_hpo_wrapper.py`) + `parity/`.
4. `metrics/` — metric tests + `parity/`.
5. `vis_tools/` — Plotly figure builders + the EvoXVision `.exv` binary format.
   Numpy/plotly only, no torch.
6. `workflows/` — `evox_etl.workflows` (EvalMonitor semantics + StdWorkflow
   monitor/history behavior).
7. `ext/` — `evox_etl_ext.autoload_ext` extension discovery/merge/idempotency
   plus the root-export + guarded-autoload tests (`test_autoload.py`).

## Gate (counts verified: 613 = 178 + 104 + 192 + 53 + 42 + 38 + 6, zero failures/errors/skips)
```
<venv>/bin/python -m pytest unit_test/etl/algorithms -q  # 178, ~3.5 min (torch-parity runs dominate)
<venv>/bin/python -m pytest unit_test/etl/operators -q   # 104, ~15 s
<venv>/bin/python -m pytest unit_test/etl/problems -q    # 192, ~100 s
<venv>/bin/python -m pytest unit_test/etl/metrics -q     # 53, ~8 s
<venv>/bin/python -m pytest unit_test/etl/vis_tools -q   # 42, ~1 s
<venv>/bin/python -m pytest unit_test/etl/workflows -q   # 38, ~14 s
<venv>/bin/python -m pytest unit_test/etl/ext -q         # 6, <1 s
```
Every `parity/` dir — plus `algorithms/mo/` and `algorithms/so/es_variants/` —
carries an `__init__.py`, so the combined run `pytest unit_test/etl -q` collects
all 613 tests with no basename collisions (the four `parity/test_parity.py` files
import as `parity.*`, `metrics.parity.*`, `problems.parity.*`,
`operators.parity.*`) and passes green (verified ~6 min).
Suite-separate runs remain useful for per-suite numbers and faster failure isolation.

## Environment
`evox_etl` and `evox_etl_ext` are PEP 420 namespace packages under `src/` (not
pip-installed), so drivers need `PYTHONPATH=src:<site-packages>`. Provision a
writable venv: `uv venv` then `uv pip install <copy-of-/home/bill/Source/etl>
pytest numpy` (etl's own checkout is read-only — install from a copy), and set
`PYTHONPATH=src:/home/bill/Source/evox/.venv/lib/python3.13/site-packages`
(that primary venv supplies plotly/torch; it is read-only).
plotly enables the figure assertions of `vis_tools/` and
`workflows/test_eval_monitor_plot.py`; without it those are `skipif`-guarded and
the plotly-missing branches still run.

## Coverage notes (durable audit findings)
- NO test under this tree uses any etl backend other than `"numpy"` on CPU: no
  `backend="iree"/"xla"`, no `Device(...)`, no cuda.
  Compiled-backend/GPU paths are exercised only by the `benchmarks/etl_vs_torch/`
  harness (torch-cuda/etl-iree-llvm-cpu/etl-iree-cuda/etl-xla-cuda), not by unit tests.
- The only skip mechanism in the tree is `skipif`-guarding of Plotly figure
  assertions (plotly present in the primary venv, so the gate run reports 0 skips);
  there are no `pytest.mark.skip`/`xfail` markers. Known-not-ported code (e.g.
  asebo's iree/xla `etl.svd` blocker) simply has no module/test to exercise it —
  see `src/evox_etl/algorithms/CONTEXT.md` + `es_variants/CONTEXT.md`.
- `evox_etl`'s own `StdWorkflow`/`EvalMonitor` (`src/evox_etl/core/workflow.py`,
  `workflows/`) ARE covered here by the `workflows/` suite (EvalMonitor shape
  policy + auxiliary-history channel + `EvalMonitor.plot`, driven directly
  through `etl.build`/`etl.run` and via real algorithms with CoDE/CSO/PSO).
  Algorithm tests still drive raw init/step via `helpers.run_generations`, and
  parity tests drive the TORCH StdWorkflow.
- Algorithms with etl-only smoke coverage and no torch parity: code/jade/ode/
  sade/shade, des/esmc/guided_es/nes/noise_reuse_es/persistent_es/snes,
  rvea/rveaa/hype. Convergence parity exists for de, pso, cma_es, open_es,
  nsga2, nsga3, moead; ars/asebo/cma_es additionally have STATE-level parity
  under injected identical noise — see `algorithms/parity/CONTEXT.md`.
- Algorithm tests construct configs via functional `make_*` constructors
  (`de_mod.make_de(...)`, `nsga2.make_nsga2(...)`, resolved at call time via
  module aliases — no direct `XConfig`/alias dataclass construction remains in
  `algorithms/`; problems tests keep direct dataclass construction by design).

## Packaging notes
- `algorithms/`, `operators/`, `problems/`, `metrics/` and their `parity/` (and
  the `mo/`/`so/`) subdirs carry an empty `__init__.py` so pytest treats them as
  distinct packages — keep them if adding new parity dirs.
  `vis_tools/` and `ext/` also carry `__init__.py` (no helper must stay top-level).
- `workflows/` deliberately has NO `__init__.py`: StdWorkflow resolves each
  component's plain functions via `type(config).__module__` + `importlib`, so the
  `aux_toy_*.py` siblings must be imported as TOP-LEVEL modules. New helper/toy
  modules there need distinctive basenames.
- Repo-root `conftest.py` shims sys.path (repo root + `src/`); each `parity/` and
  the problems/metrics/vis_tools/workflows suites also carry idempotent
  relocate-ready conftest shims.
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
