# benchmarks/etl_vs_torch — torch evox vs evox_etl comparison

## Intent
Benchmarks comparing the torch OOP evox (`src/evox/`) against the functional
`evox_etl` (`src/evox_etl/`) on CPU and GPU, small → large scale, plus a code
style/cleanliness comparison report. See `../../src/evox_etl/DESIGN.md` §7.

## Contents
- `bench_common.py` — sys.path shim, case tables, numpy metric helpers
  (hv/igd + seeded DTLZ ref fronts), parity helpers, GPU env recipe,
  JSON record schema. Pure stdlib + numpy (no torch/etl imports).
- `bench_so.py` / `bench_mo.py` — per-backend runners (torch eager, etl
  numpy/iree/xla), resilient per-case error records.
- `run.py` — CLI dispatcher: `--backend --suite {so,mo} [--gpu N]
  [--cases ...] [--gens-override N] [--list]`; sets the GPU env BEFORE
  backend imports.
- `BENCHMARK_RESULTS.md` — the comparison report: full SO/MO result tables,
  compile-time and speedup tables, caps & limits, parity verdicts, key
  numbers, reproduction. **All numbers come from `results/*.json`.**
- `style_comparison.md` — code style/cleanliness pros & cons (torch OOP vs
  etl functional).
- `results/` — 12 committed JSONs (`{so,mo}_{backend}.json` × 6 backends),
  288 records with parity verdicts vs torch-cpu written in. Schema, metric
  definitions, timing semantics and quirks are documented in
  `results/README.md`.

## How to run
Venv: `/mnt/local-ssd/bchuang/evox/.venv/bin/python` (torch 2.6.0+cu124,
Python 3.11.2). From this directory:

```
python run.py --backend torch-cpu      --suite so     # and --suite mo
python run.py --backend torch-cuda     --suite so --gpu 0
python run.py --backend etl-numpy      --suite so
python run.py --backend etl-iree-llvm-cpu --suite so
python run.py --backend etl-iree-cuda  --suite so --gpu 0
python run.py --backend etl-xla-cuda   --suite so --gpu 0
```

`--gpu N` applies the GPU recipe before any torch/etl import
(`bench_common.ensure_gpu_env`): `CUDA_VISIBLE_DEVICES=<id>`, prepend
`/mnt/local-ssd/bchuang/cudnn-xla/lib` to `LD_LIBRARY_PATH`,
`XLA_FLAGS=--xla_gpu_cuda_data_dir=/home/bchuang/xla_cuda_data`,
`ETL_PJRT_PLUGIN=<venv>/jax_plugins/xla_cuda12/xla_cuda_plugin.so`;
etl-xla-cuda additionally LD_PRELOADs cuDNN 9.8 and re-execs the process
(marker-guarded). Use `--cases` / `--gens-override` / `--out <tmp path>` for
subsets and smoke runs — never write smoke outputs into `results/`.

## Current results status
- **Numbers exist for:** SO on all six backends (36 cases each, no errors);
  MO full on torch-cpu, torch-cuda, etl-numpy, and etl-xla-cuda (12 cases
  each, 100 gens — xla-cuda became a full matrix after the NSGA3 fix);
  iree-llvm-cpu / iree-cuda 12/12 error records.
- **etl-numpy caps:** SO 1000x50 → 50 gens, 10000x100 → 10 gens
  (interpreter too slow). The MO cap is removed: Pareto fronts are now
  extracted per-generation in `bench_mo.py` (`_etl_pf_fitness`, mirroring
  `_torch_pf_fitness`), replacing the O(n²) whole-history ranking that made
  100 gens infeasible — verified point-for-point identical.
- **etl-GPU MO blockers:** the DTLZ `cumprod`/`flip` v1-export issue and the
  NSGA3 `matrix_rank`/`solve` export rejection are both FIXED (nsga3.py now
  uses eigh-based full-rank guard + normal-equations solve). Remaining: (1)
  iree-compile segfault (`Error code: -11`, stack in `libIREECompiler.so`) on
  ALL MO programs — 23 cells: iree-cuda 12/12, iree-llvm-cpu 11/12 (NSGA3
  now reaches the compiler post-fix and crashes identically to NSGA2/MOEAD —
  an upstream iree while-loop compiler bug, proven by the unmodified
  NSGA2/MOEAD trigger; do not attempt to fix iree); (2)
  MOEAD/DTLZ2/100x3x10 on iree-llvm-cpu dies at runtime (`ref is null`,
  `hal.buffer_view.create`).
- **CMA-ES `c_c` framework bug is FIXED** in both torch and etl (Hansen
  canonical formula); all 9 CMA-ES SO cells on all 6 backends were re-run.
- **CMAES/Ackley/100x10 now converges on all four etl backends** (best
  0.00055–0.0037, matching torch-cpu 0.0036; iree-llvm-cpu parity ok at rel
  0.007). This required replicating torch's 1-D `p_c @ p_c.T`
  scalar-dot-product quirk (isotropic c_1 inflation) instead of the
  canonical rank-one outer product — see
  `src/evox_etl/algorithms/so/es_variants/`. Large-scale Ackley cells
  (1000x50/10000x100) still scatter on unconverged runs (torch-cuda
  ≈20.6/20.4; etl compiled backends rel 0.44–1.89) — RNG-stream variance,
  not a stall.
- **CMAES slow path on iree is characterized, not a defect:** per-step
  covariance `etl.eigh` lowers to an iree cyclic-Jacobi while-loop
  composition (no native stablehlo.eigh in stablehlo 1.0 / iree 3.11.0),
  O(n^4). Probes: standalone iree-cuda eigh 39.0 / 1317 / 5778 ms @ dims
  10/50/100 vs 0.087 ms no-eigh control; llvm-cpu dim-100 23676 ms vs 0.157;
  real-harness iree-cuda gens=10 matched (39.5 / 1036.6 / 5779.3 ms/step);
  xla-cuda native eigh 387.3 ms/step (15-18× faster). All iree CMAES cells
  carry a JSON note; upstream doc:
  `/mnt/local-ssd/bchuang/etl/etl/backends/stablehlo/CONTEXT.md`.
- **Parity (vs torch-cpu, 10% rel tolerance, near-zero rule):** SO 94 ok /
  62 not-ok / 24 skip (etl-numpy gens caps); MO 12 ok / 24 not-ok / 24 skip
  (backend errors). etl compiled backends are bit/last-ulp identical to
  each other; not-ok cells are RNG-stream scatter on unconverged runs
  (small-scale 100x10 Rastrigin/DE/OpenES, large-scale Ackley 1000x50/
  10000x100, MO DTLZ1 incl. xla-cuda NSGA3 rel 16.636/0.528 — see
  `BENCHMARK_RESULTS.md`).
- **etl-xla-cuda is device-resident and measured at torch-cuda speed
  (2026-09-07, idle A6000 #6, seed 42, 100 gens, compile excluded):**
  PSO/Sphere 1000x50 0.896, 10000x100 0.988/1.045 ms/step; DE/Rastrigin
  1000x50 0.799, 10000x100 0.776/0.843 ms/step — 1.16-1.45× faster than
  torch-cuda (1.036-1.217 ms/step, same-day), convergence identical to the
  committed records. This required both landed fixes: the harness mapping
  (`bench_common.etl_backend_spec` → `("xla", "cuda:0")`) and the workflow
  backend activation before device placement (commit 1a6fef1c,
  `workflow.py:229-244`). The committed xla SO cells (5.73/17.0 ms/step)
  are host-staging-bound artifacts of the old `("xla", None)` mapping —
  stale, see the caveat note in `BENCHMARK_RESULTS.md`.
- **Both iree SO JSONs are fully refreshed under iree 3.11.0 on healthy
  hardware** (36/36 records each: `so_etl-iree-cuda.json` commit 2770e1fc on
  GPU 0, `so_etl-iree-llvm-cpu.json` commit ba895503 under the fixed etl).
  The old broken-GPU-era big-scale values (35.70 / 3482 ms/step @1000x50 /
  @10000x100) are gone — git history preserves them. The refreshed CMAES
  cells are the slow-path exception (per-step eigh lowering, see the CMAES
  slow-path bullet above), explicitly noted in the JSONs.
- Details, tables and key numbers: see `BENCHMARK_RESULTS.md`.

## GPU performance path — issues and fixes (measurement-verified, current state)
- **The old etl-xla-cuda slowness was the harness mapping + placement bug,
  now FIXED:** `bench_common.etl_backend_spec` maps etl-xla-cuda to
  `("xla", "cuda:0")` (commit 8202692f; etl HEAD f2f50a7 supports
  device-resident `Device("cuda", N)` executables), and `StdWorkflow` now
  activates the workflow's own backend adapter before placing state on
  device (`workflow.py:229-244`, commit 1a6fef1c) — replacing the manual
  pre-activation (`etl.backends.get("xla")` before `wf.init`) that was
  previously required because `Tensor.to(cuda)` dispatches through a
  process-global, last-wins device-transfer slot (lazy iree thunk from
  `etl/backends/__init__.py`; the xla adapter overwrites it on activation).
- **Old committed xla SO numbers are host-staging-bound:** a `("xla", None)`
  device makes the workflow CPU-kind (`workflow.py:45-48`) and the xla
  adapter host-stages EVERY input every step (`.numpy()` + `buffer_from_host`
  per call) — measured PSO/Sphere on a healthy GPU: 6.23 / 15.32 ms/step
  @1000x50 / @10000x100, reproducing the committed 5.73 / 17.0. Post-fix
  device-resident measurements (2026-09-07, idle A6000): 0.90 / 0.99-1.05
  ms/step @1000x50 / @10000x100 PSO/Sphere and 0.80 / 0.78-0.84 DE/Rastrigin,
  i.e. 1.16-1.45× faster than torch-cuda — see the caveat note in
  `BENCHMARK_RESULTS.md`.
- **iree 3.9.0 CUDA segfault is gone under iree ≥ 3.11.0:** the earlier
  deterministic step-1 crash on the harness path (bisected to `workflow.step`
  retaining `self._state = state`, now `workflow.py:323`) no longer occurs —
  etl crash repro passed 4/4 and both full iree SO suites ran clean on
  3.11.0. The remaining iree blocker is only the MO while-loop compile bug
  (see Current results status; out of scope — xla remains the MO compiled
  backend).
- **Per-step host copies are NOT the bottleneck once device-resident:** history D2H readback (`.to(cpu).numpy()` of `latest_fitness`, `workflow.py:413-416`; `full_sol/pop_history` default off, `eval_monitor.py:52-54`) ≈ 0.09 ms/step; same-device `tree_map .to` no-op (`workflow.py:317-318`) ≈ 0.08 ms/step. The monitor's in-graph topk/gather is the main marginal cost at scale (≈1.5-2.3 ms of the 2.4-3.05 ms/step @10000x100 iree-cuda). Enabling `full_sol_history`/`full_pop_history` WOULD copy (pop,dim) per step — keep off in benchmarks.
- **Timing fairness:** the etl path times without an explicit device sync (`bench_so.py:147-156`; the torch path syncs at 76-81), but iree/xla `run` invokes are host-blocking, so steps cannot overlap — per-step timing is fair; the 2 warmup steps (`bench_common.WARMUP_STEPS`) exercise the same `step()` path and absorb first-call costs.

## Notes for agents
- Result JSON schema and field semantics are documented in
  `results/README.md` — read it before touching the JSONs.
- **The runners OVERWRITE `results/{suite}_{backend}.json` with only the
  cases they run.** Partial re-runs must use `--out <tmp>` and then merge
  with `retag_parity.py --partial <tmp> [--cap-note "..."] [--set-note
  CASE=TEXT] [--note-override CASE=TEXT] --apply`, which merges by `case_id`
  and re-derives parity verdicts per the committed conventions (machine-zero
  rule, skip reasons, RNG-variance notes). Never hand-edit verdicts —
  recompute with the script.
- `parity` blocks compare each non-torch-cpu record against the torch-cpu
  record with the same `case_id`; `ok` is `true`/`false`/`null` (null only
  for skips: backend error, baseline error, or gens cap). Near-zero pairs
  (both ≤ 1e-4) are ok by rule; the full semantics are stated in
  `BENCHMARK_RESULTS.md` §Parity verdicts.
- Never ship wrong numbers: every cell in `BENCHMARK_RESULTS.md` must trace
  back to a `results/*.json` record; error/capped records must be marked
  (`err`, `@Ng`) instead of shown as plain values.
- Known harness quirks (torch MO default-device workaround, per-generation
  PF filter instead of `EvalMonitor.get_pf_fitness`, xla `device=None`
  staging, MOEAD pop_size overwritten by Das-Dennis count, NSGA3
  non-reproducibility across GPU runs) are recorded in `results/README.md`
  and `BENCHMARK_RESULTS.md` — do not re-derive them.
