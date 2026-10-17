# benchmarks/etl_vs_torch — torch evox vs evox_etl comparison

## Intent
Benchmarks comparing the torch OOP evox (`src/evox/`) against the functional
`evox_etl` (`src/evox_etl/`) on CPU and GPU, small → large scale, plus a code
style/cleanliness comparison report. See `../../src/evox_etl/DESIGN.md` §7.

## Contents
- `bench_common.py` — sys.path shim, case tables, numpy metric helpers
  (hv/igd + seeded DTLZ ref fronts), parity helpers, NVIDIA **and** AMD GPU
  env recipes, backend id tables, etl compile-retry helper, JSON record
  schema. Pure stdlib + numpy (no torch/etl imports).
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
- `results/` — committed JSONs: 12 CUDA/A6000 files (`{so,mo}_{backend}.json`
  × 6 backends, 288 records) plus 8 MI300X/ROCm files —
  `{so,mo}_torch-rocm.json`, `{so,mo}_etl-xla-rocm.json`,
  `{so,mo}_etl-iree-rocm.json`, `{so,mo}_etl-iree-llvm-cpu_mi300x.json` —
  every record carrying a parity verdict vs `results/{suite}_torch-cpu.json`.
  Schema, metric definitions, timing semantics and quirks live in
  `results/README.md`.
- `results/rocm_prefix_baseline.json` — the pre-fix ROCm matrix (SO + MO, all
  five MI300X columns incl. `etl-numpy`) reconstructed from the earlier
  round's scratch table dumps, with a `provenance` block; it is the traceable
  source for every pre-fix ROCm number and for the REUSED `etl-numpy` MI300X
  numbers.

## Backends

`bench_common.ALL_BACKENDS` (id → etl `(backend, device)`; torch ids use
`bench_common.torch_device`):

| id | vendor | etl spec / torch device |
|---|---|---|
| `torch-cpu` | — | torch `"cpu"` |
| `torch-cuda` | nvidia | torch `"cuda"` |
| `torch-rocm` | rocm | torch `"cuda"` (ROCm torch exposes HIP through `torch.cuda`; there is no `"rocm"` torch device) |
| `etl-numpy` | — | `("numpy", None)` |
| `etl-iree-llvm-cpu` | — | `("iree", None)` |
| `etl-iree-cuda` | nvidia | `("iree", "cuda:0")` |
| `etl-iree-rocm` | rocm | `("iree", "rocm:0")` |
| `etl-xla-cuda` | nvidia | `("xla", "cuda:0")` |
| `etl-xla-rocm` | rocm | `("xla", "rocm:0")` |

`CUDA_BACKENDS` / `ROCM_BACKENDS` / `PJRT_BACKENDS` (the two xla ids) /
`TORCH_BACKENDS` drive env selection, the `--gpu` requirement and the exit
path. Case tables, seeds, timed region, JSON schema and parity logic are
backend-independent.

All nine ids are wired AND measured: the three `*-rocm` ids have MI300X /
ROCm 7.1 columns in `results/`, and the CPU column was refreshed on that host
too (`etl-iree-llvm-cpu_mi300x`).

**Host distinction (do not mix columns):** the committed
`results/{suite}_{backend}.json` files for `etl-numpy`, `etl-iree-llvm-cpu`
(no suffix), `torch-cpu`, `torch-cuda`, `etl-iree-cuda` and `etl-xla-cuda`
were measured on a DIFFERENT host (A6000 / CUDA) and still back the CUDA
sections of `BENCHMARK_RESULTS.md` — never overwrite them with MI300X
numbers. Measurements taken on the MI300X host use the natural `*-rocm` file
names, or a `_mi300x` suffix for CPU-column refreshes
(`results/{so,mo}_etl-iree-llvm-cpu_mi300x.json`).

## How to run
NVIDIA venv: `/mnt/local-ssd/bchuang/evox/.venv/bin/python` (torch
2.6.0+cu124, Python 3.11.2). ROCm venv:
`/tmp/genesis/task_3858dfe003526650/etlbench-venv/bin/python` (torch
2.13.0+rocm7.1, Python 3.14). From this directory:

```
python run.py --backend torch-cpu      --suite so     # and --suite mo
python run.py --backend torch-cuda     --suite so --gpu 0
python run.py --backend etl-numpy      --suite so
python run.py --backend etl-iree-llvm-cpu --suite so
python run.py --backend etl-iree-cuda  --suite so --gpu 0
python run.py --backend etl-xla-cuda   --suite so --gpu 0
python run.py --backend torch-rocm     --suite so --gpu 0     # ROCm venv
python run.py --backend etl-iree-rocm  --suite so --gpu 0     # ROCm venv
python run.py --backend etl-xla-rocm   --suite so --gpu 0     # ROCm venv
```

`--gpu N` applies the vendor recipe before any torch/etl import through
`bench_common.ensure_backend_env(backend, N)`, which re-execs the process
once (marker `EVOX_BENCH_GPU_ENV`, idempotent, keeps `sys.argv`):

- **NVIDIA** (`set_gpu_env`): `CUDA_VISIBLE_DEVICES=<id>`, prepend
  `/mnt/local-ssd/bchuang/cudnn-xla/lib` to `LD_LIBRARY_PATH`,
  `XLA_FLAGS=--xla_gpu_cuda_data_dir=/home/bchuang/xla_cuda_data`,
  `ETL_PJRT_PLUGIN=<venv>/jax_plugins/xla_cuda12/xla_cuda_plugin.so`;
  etl-xla-cuda additionally LD_PRELOADs cuDNN 9.8.
- **AMD ROCm** (`set_rocm_env`): `HIP_VISIBLE_DEVICES=<id>` +
  `ROCR_VISIBLE_DEVICES=<id>`, no cuDNN preload, no
  `--xla_gpu_cuda_data_dir`. The xla ids (etl-xla-rocm) additionally set
  `ETL_PJRT_PLUGIN=<rocm venv>/jax_plugins/xla_rocm7/xla_rocm_plugin.so`
  (an already-set `ETL_PJRT_PLUGIN` wins) and append
  `XLA_FLAGS=--xla_gpu_enable_command_buffer=` — HIP graph command buffers
  SIGSEGV inside `PJRT_Executable_Execute`.

Use `--cases` / `--gens-override` / `--out <tmp path>` for subsets and smoke
runs — never write smoke outputs into `results/`.

On the MI300X host the ROCm venv writes `results/{suite}_torch-rocm.json`,
`results/{suite}_etl-iree-rocm.json` and `results/{suite}_etl-xla-rocm.json`;
an `etl-iree-llvm-cpu` re-run here must be sent with `--out
results/{suite}_etl-iree-llvm-cpu_mi300x.json` so the A6000 file is not
overwritten. Runs are serial and foreground — the ROCm device is
single-tenant.

## Robustness on the etl/GPU paths
- **etl compile retry:** `bench_common.init_with_retry` (used by both
  runners) builds a FRESH workflow and calls `init(seed)` up to
  `ETL_COMPILE_ATTEMPTS` (4) times with a 1 s backoff, keeping the first
  successful artifact and its wall time as `compile_time_s` (only that
  attempt is timed). It is applied to every etl backend (harmless where
  compiles are deterministic) because the iree MO compile fails
  nondeterministically (upstream `Stream::AffinityAnalysis` recursion). When
  every attempt fails the last exception is re-raised and `run_one` writes
  the usual per-case error record. `KeyboardInterrupt`/`SystemExit` are never
  retried. A hard compiler SIGSEGV still terminates the process (Python-level
  retry cannot catch signals).
- **PJRT exit handling:** xla processes can abort during PJRT teardown AFTER
  the work is done. `save_results` flushes + fsyncs the JSON, then
  `bench_common.finish_run(backend)` flushes stdio and leaves via
  `os._exit(0)` for `PJRT_BACKENDS` — a completed run is never reported as a
  crash. Non-PJRT backends return and exit normally; a genuinely failed run
  raises before `finish_run` (per-case failures are already error records).

## Current results status
- **ROCm host (AMD Instinct MI300X, one VF, ROCm 7.1) columns:** SO —
  `torch-rocm` 36 records (9 CMA-ES `err`), `etl-xla-rocm` 36/36,
  `etl-iree-rocm` 36/36, `etl-iree-llvm-cpu_mi300x` 36/36 (CMA-ES 10000x100
  at the full 100 gens — no cap); MO — `torch-rocm` 12/12, `etl-xla-rocm`
  12/12, `etl-iree-rocm` 12/12 `err` (iree-compile SIGSEGV) and
  `etl-iree-llvm-cpu_mi300x` 3/12 run + 9/12 `err` (the three successes are
  `MOEAD/DTLZ1/100x3x10`, `MOEAD/DTLZ1/1000x3x30`,
  `MOEAD/DTLZ2/1000x3x30`).
- **`etl-numpy` MI300X numbers are REUSED, not re-measured:** the numpy
  backend is unaffected by the etl device-path fixes, so its ROCm-host cells
  come from `results/rocm_prefix_baseline.json` and must be labelled reused.
- **`torch-rocm` CMA-ES does not exist:** all 9 CMA-ES cells are `err`
  (`UncapturedHigherOrderOpError ... torch.cond`), a torch 2.13
  incompatibility that also reproduces on torch-cpu — the torch CMA-ES
  reference is unavailable on this host; do not chase it.
- **ROCm headline numbers (ms/step, seed 42, 100 gens):** CMA-ES 10000x100 is
  43× faster on `etl-iree-rocm` (4127 → 94.6), 69× on `etl-xla-rocm`
  (1256 → 18.2) and 4.8× on `etl-iree-llvm-cpu_mi300x` (10578 @22 gens →
  2182 @100 gens); non-CMA-ES SO — PSO/Sphere/10000x100 on `etl-iree-rocm`
  2.61 → 0.55; MO on `etl-xla-rocm` 0.64–53.8 ms/step, i.e. 3.5–137× faster
  than `torch-rocm`.
- **OpenES is the one SO family where `torch-rocm` wins:** the etl compiled
  backends carry a ~0.35–0.45 ms/step host-side floor (0.38–0.68 measured vs
  0.21–0.33 for torch-rocm), which dominates steps this small.
- **CUDA host (A6000) columns, unchanged:** SO on all six CUDA-era backends
  (36 cases each, no errors); MO full on torch-cpu, torch-cuda, etl-numpy and
  etl-xla-cuda (12 cases each, 100 gens); iree-llvm-cpu / iree-cuda 12/12
  error records.
- **etl-numpy caps:** SO 1000x50 → 50 gens, 10000x100 → 10 gens
  (interpreter too slow). The MO cap is removed: Pareto fronts are extracted
  per-generation in `bench_mo.py` (`_etl_pf_fitness`, mirroring
  `_torch_pf_fitness`), replacing the O(n²) whole-history ranking that made
  100 gens infeasible — verified point-for-point identical.
- **iree MO compile blocker (current, no fix attempted):** iree-compile
  SIGSEGV (`Error code: -11`, stack in `libIREECompiler.so`) on ALL MO
  programs — an upstream iree while-loop compiler bug, proven by the
  unmodified NSGA2/MOEAD trigger. Counts: `etl-iree-cuda` 12/12,
  `etl-iree-rocm` 12/12 (deterministic: 4 in-process + 4 fresh-process
  attempts per case), `etl-iree-llvm-cpu` 11/12 (A6000) and
  `etl-iree-llvm-cpu_mi300x` 9/12 (MO compile is nondeterministic on
  llvm-cpu, so the 3 successes needed process-level retries — the documented
  recipe). `etl-xla-*` is the working compiled MO backend. Separately,
  `MOEAD/DTLZ2/100x3x10` on A6000 llvm-cpu passes compile and dies at
  runtime (`ref is null`, `hal.buffer_view.create`).
- **CMAES/Ackley/100x10 converges on all four etl backends** (best
  0.00055–0.0037, matching torch-cpu 0.0036; iree-llvm-cpu parity ok at rel
  0.007), which requires replicating torch's 1-D `p_c @ p_c.T`
  scalar-dot-product quirk (isotropic c_1 inflation) instead of the
  canonical rank-one outer product — see
  `src/evox_etl/algorithms/so/es_variants/`. Large-scale Ackley cells
  (1000x50/10000x100) still scatter on unconverged runs (torch-cuda
  ≈20.6/20.4; etl compiled backends rel 0.44–1.89) — RNG-stream variance,
  not a stall.
- **CMAES is the slowest SO family on every compiled backend:** per-step
  covariance `etl.eigh` lowers to an iree cyclic-Jacobi while-loop
  composition (no native stablehlo.eigh in stablehlo 1.0 / iree 3.11.0),
  O(n^4). Probes: standalone iree-cuda eigh 39.0 / 1317 / 5778 ms @ dims
  10/50/100 vs 0.087 ms no-eigh control; llvm-cpu dim-100 23676 ms vs 0.157;
  real-harness iree-cuda gens=10 matched (39.5 / 1036.6 / 5779.3 ms/step);
  xla-cuda native eigh 387.3 ms/step. Current dim-100 magnitudes: 94.6
  ms/step on iree-rocm, 18.2 on xla-rocm, 2182 on llvm-cpu-mi300x. All iree
  CMAES cells carry a JSON note; upstream doc:
  `/mnt/local-ssd/bchuang/etl/etl/backends/stablehlo/CONTEXT.md`.
- **Parity (10% rel tolerance, near-zero rule):** every non-torch-cpu record
  is compared against the `results/{suite}_torch-cpu.json` record with the
  same `case_id`; `ok` is `true`/`false`/`null` (null only for skips: backend
  error, baseline error, gens cap). etl compiled backends are bit/last-ulp
  identical to each other; not-ok cells are RNG-stream scatter on unconverged
  runs (small-scale 100x10 Rastrigin/DE/OpenES, large-scale Ackley
  1000x50/10000x100, MO DTLZ1 incl. xla-cuda NSGA3 rel 16.636/0.528) — see
  `BENCHMARK_RESULTS.md` for the per-column verdict tables.
- **etl-xla-cuda is device-resident and measured at torch-cuda speed
  (2026-09-07, idle A6000 #6, seed 42, 100 gens, compile excluded):**
  PSO/Sphere 1000x50 0.896, 10000x100 0.988/1.045 ms/step; DE/Rastrigin
  1000x50 0.799, 10000x100 0.776/0.843 ms/step — 1.16-1.45× faster than
  torch-cuda (1.036-1.217 ms/step, same-day), convergence identical to the
  committed records. This relies on both landed fixes: the harness mapping
  (`bench_common.etl_backend_spec` → `("xla", "cuda:0")`) and the workflow
  backend activation before device placement (commit 1a6fef1c,
  `workflow.py:229-244`). The committed xla SO cells (5.73/17.0 ms/step)
  are host-staging-bound artifacts of the old `("xla", None)` mapping —
  stale, see the caveat note in `BENCHMARK_RESULTS.md`.
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
- **Timing fairness:** the etl path times without an explicit device sync (`bench_so.py:159-162`; the torch path syncs at 75-80), but iree/xla `run` invokes are host-blocking, so steps cannot overlap — per-step timing is fair; the 2 warmup steps (`bench_common.WARMUP_STEPS`) exercise the same `step()` path and absorb first-call costs.

## Notes for agents
- **ROCm runs:** use the ROCm venv
  (`/tmp/genesis/task_3858dfe003526650/etlbench-venv/bin/python`, Python 3.14,
  torch 2.13.0+rocm7.1, etl 0.1.0, `evox_etl` imported from the repo `src/`)
  and always pass `--gpu N` (`HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES`
  are set and the process re-execs before any import). The ROCm device is
  single-tenant — never run two ROCm jobs at once, and run them serially in
  the foreground. Write ROCm results to `results/{suite}_torch-rocm.json`,
  `results/{suite}_etl-iree-rocm.json`, `results/{suite}_etl-xla-rocm.json`;
  an `etl-iree-llvm-cpu` re-run on this host goes to
  `results/{suite}_etl-iree-llvm-cpu_mi300x.json` via `--out` (the default
  name would clobber the A6000 file).
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

### Known quirks (ROCm/MI300X host)
- **`torch-rocm` teardown abort:** the process can abort in the HIP teardown
  path (`c10::cuda::SetDevice` inside a `TensorImpl` destructor) AFTER the
  JSON is written. Read stdout / the JSON as the source of truth — it is not
  a failed run and must not be re-run or re-recorded as an error.
- **`torch-rocm` MO timing variance:** full sweeps differ (first sweep 5–9%
  slower, one NSGA3 cell 32% slower) and NSGA3 does not reproduce
  point-for-point across GPU runs. Quote numbers from the committed JSON,
  never from a re-run.
- **Torch MO on this box** uses the harness's existing default-device
  workaround in `bench_mo.py` — do not re-derive it.
- **AMD xla recipe:** `ETL_PJRT_PLUGIN` → the venv's
  `jax_plugins/xla_rocm7/xla_rocm_plugin.so`, plus
  `XLA_FLAGS=--xla_gpu_enable_command_buffer=` (HIP command buffers SIGSEGV
  inside `PJRT_Executable_Execute`). `torch-rocm` has no `"rocm"` torch
  device — `torch.cuda` IS HIP.
- **PJRT/xla teardown on ROCm** can also abort after all work is done; the
  harness flushes + fsyncs the JSON and exits via `os._exit(0)` for
  `PJRT_BACKENDS` (see Robustness above), so a completed run is never
  reported as a crash.
- **`torch-rocm` has no CMA-ES reference** (`UncapturedHigherOrderOpError ...
  torch.cond`, reproducible on torch-cpu): those 9 SO records stay `err` and
  their parity stays a skip.
