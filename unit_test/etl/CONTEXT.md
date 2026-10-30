# unit_test/etl — tests for the functional evox_etl package

## Intent
pytest suite mirroring `src/evox_etl/` (functional EvoX on evoxc). Seven sub-suites
plus the root-level torch-free release gate:
1. `algorithms/` — algorithm smoke tests + converted operator-shim tests
   (canonical `evox_etl.operators.*` imports; ETL-only, numpy expectations).
   Includes the virtual-ES / VirtualLoRA-ES smoke tests and the virtual
   end-to-end convergence tests under `so/es_variants/`.
   Ported-upstream-fix regressions: initial-population bounds/uniformity for
   CoDE/SaDE/SHADE (`so/de_variants/test_init_bounds.py`), SHADE success-history
   adaptation (`so/de_variants/test_shade.py`), NES default `pop_size` and
   GuidedES `subspace_dims < dim` (`so/es_variants/test_nes.py`,
   `so/es_variants/test_guided_es.py`).
2. `operators/` — operator property tests (keyed random ops + the keyless
   samplers; numpy only).
   Includes the uniform `DE_binary_crossover` mask-rate regression
   (`test_crossover.py`) and the 1-D tournament index shape / large-tournament
   winner regressions (`test_selection.py`).
3. `problems/` — numerical problem tests + the neuroevolution virtual-problem
   suite (`test_virtual_problem.py`) + the host-side HPO-wrapper suite
   (`test_hpo_wrapper.py`).
4. `metrics/` — metric tests; includes the hypervolume
   reference-point clipping regressions (`hv` / `each_cube_monte_carlo_hv`).
5. `vis_tools/` — Plotly figure builders + the EvoXVision `.exv` binary format.
   Numpy/plotly only, no torch.
6. `workflows/` — `evox_etl.workflows` (EvalMonitor semantics + StdWorkflow
   monitor/history behavior).
7. `ext/` — `evox_etl_ext.autoload_ext` extension discovery/merge/idempotency
   plus the root-export + guarded-autoload tests (`test_autoload.py`).
8. `test_no_torch.py` (root) — release gate: a FRESH interpreter imports
   `evox_etl` and runs a small PSO-on-Sphere optimization, then asserts no
   `torch` module entered `sys.modules`.

## Gate (counts verified: 552 = 182 + 86 + 144 + 50 + 42 + 41 + 6 + 1)
```
PYTHONPATH=src <venv>/bin/python -m pytest unit_test/etl/algorithms -q      # 182, ~80 s
PYTHONPATH=src <venv>/bin/python -m pytest unit_test/etl/operators -q       #  86, ~5 s
PYTHONPATH=src <venv>/bin/python -m pytest unit_test/etl/problems -q        # 144, ~30 s
PYTHONPATH=src <venv>/bin/python -m pytest unit_test/etl/metrics -q         #  50, ~5 s
PYTHONPATH=src <venv>/bin/python -m pytest unit_test/etl/vis_tools -q       #  42, ~1 s
PYTHONPATH=src <venv>/bin/python -m pytest unit_test/etl/workflows -q       #  41, ~5 s
PYTHONPATH=src <venv>/bin/python -m pytest unit_test/etl/ext -q             #   6, <1 s
PYTHONPATH=src <venv>/bin/python -m pytest unit_test/etl/test_no_torch.py -q #  1, <1 s
```
The combined run `PYTHONPATH=src <venv>/bin/python -m pytest -q` from the
repository root collects all 552 tests with no basename collisions and passes
with 0 errors and 0 skips in ~2 min (`pyproject.toml` sets
`testpaths = ["unit_test/etl"]`). Every subdirectory that carries an
`__init__.py` (`algorithms/mo/`, `algorithms/so/es_variants/`, `operators/`,
`problems/`, `metrics/`, `vis_tools/`, `ext/`) imports its modules with the
package prefix, which is what keeps same-basename files distinct.
Suite-separate runs remain useful for per-suite numbers and faster failure isolation.

## Environment
The compiler the port and these tests run on is `evoxc` (distribution `evoxc`
0.1.0; traced surface `evoxc`, `evoxc.core`, `evoxc.numpy`, `evoxc.random`,
`evoxc.io`, `evoxc.ops`).
`evox_etl` and `evox_etl_ext` are PEP 420 namespace packages under `src/` (not
pip-installed when working from a checkout), so drivers need
`PYTHONPATH=src:<site-packages>`. Provision a writable venv: `uv venv` then
`uv pip install evoxc pytest numpy` — the `evoxc` distribution is publish-ready
but not yet on a package index, so install it from a source checkout (copy it
somewhere writable if the checkout is read-only).
torch is NOT part of this environment: the 2.0 release ships only
`evoxc` + `numpy`, and no test under this tree may pull torch in.
plotly enables the figure assertions of `vis_tools/` and
`workflows/test_eval_monitor_plot.py`; without it those are `skipif`-guarded and
the plotly-missing branches still run.

## Coverage notes (durable audit findings)
- NO test under this tree uses any evoxc backend other than `"numpy"` on CPU: no
  `backend="iree"/"xla"`, no `Device(...)`, no cuda.
  Compiled-backend/GPU paths are exercised only by the `benchmarks/etl_vs_torch/`
  harness (torch-cuda/etl-iree-llvm-cpu/etl-iree-cuda/etl-xla-cuda), not by unit tests.
- The only skip mechanism in the tree is `skipif`-guarding of Plotly figure
  assertions (plotly present in the dev venv, so the gate run reports 0 skips);
  there are no `pytest.mark.skip`/`xfail` markers. Known-not-ported code (e.g.
  asebo's iree/xla `evoxc.svd` blocker) simply has no module/test to exercise it —
  see `src/evox_etl/algorithms/CONTEXT.md` + `es_variants/CONTEXT.md`.
- `evox_etl`'s own `StdWorkflow`/`EvalMonitor` (`src/evox_etl/core/workflow.py`,
  `workflows/`) ARE covered here by the `workflows/` suite (EvalMonitor shape
  policy + auxiliary-history channel + `EvalMonitor.plot`, driven directly
  through `evoxc.build`/`evoxc.run` and via real algorithms with CoDE/CSO/PSO).
  Algorithm tests drive raw init/step via `helpers.run_generations`.
- Algorithm tests construct configs via functional `make_*` constructors
  (`de_mod.make_de(...)`, `nsga2.make_nsga2(...)`, resolved at call time via
  module aliases — no direct `XConfig`/alias dataclass construction remains in
  `algorithms/`; problems tests keep direct dataclass construction by design).

## Packaging notes
- `operators/`, `problems/`, `metrics/`, `vis_tools/`, `ext/` carry an
  `__init__.py`, as do `algorithms/mo/` and `algorithms/so/es_variants/`; pytest
  then imports those test modules with their package prefix
  (`es_variants.test_virtual_es`, …), which is what keeps same-basename files
  from colliding. Keep the markers when adding new package dirs.
- `algorithms/`, `algorithms/so/`, `algorithms/so/de_variants/`,
  `algorithms/so/pso_variants/` and `workflows/` deliberately have NO `__init__.py`,
  so their test modules import as TOP-LEVEL names — new files there need
  distinctive basenames.
- `workflows/` in particular must stay flat: StdWorkflow resolves each component's
  plain functions via `type(config).__module__` + `importlib`, so the `aux_toy_*.py`
  siblings have to remain top-level modules.
- Repo-root `conftest.py` shims sys.path (repo root + `src/`); the
  problems/metrics/vis_tools/workflows suites (and `algorithms/`) also carry
  idempotent relocate-ready conftest shims.
- Test pattern: `evoxc.build(fn, *specs, backend="numpy")` + `evoxc.run(exe, *args)`;
  scalar tensor inputs are 0-d ndarrays (numpy scalars rejected at run boundary);
  statics re-passed at run in signature order; results need `.numpy()`.

## Known operator deviations (reported to root agent, tests adapted)
- Canonical `apd_fn` gathers with `relu(x)` where the reference uses
  `norm_obj[x]` (negative-index wrap) — latent in `ref_vec_guided`; details in
  `operators/CONTEXT.md` and `algorithms/CONTEXT.md`.
- Canonical `DE_differential_sum` second output dtype varies (int32 replace=True,
  int64 replace=False) — upstream-faithful promotion.

See `../../src/evox_etl/DESIGN.md` §6 for the testing strategy.
