# results/ — 1.x torch EvoX vs functional 2.0 benchmark outputs

> **Historical evidence (1.x torch era).** Every JSON in this directory is a
> frozen measurement record taken while the PyTorch 1.x reference
> implementation still existed, during the 1.4 → 2.0 refactor; the
> files are preserved byte-for-byte and are not re-measured.
> The `*_torch-*.json` files (`torch-cpu`, `torch-cuda`, `torch-rocm`) are kept
> as historical references only — those backend ids no longer exist in the 2.0
> harness.
> The `evox-*` filenames are frozen keys that name the `evoxc`-backed
> implementations (`evox-numpy` = numpy backend, `evox-iree-*` = iree backend,
> `evox-xla-*` = xla backend).

## Naming convention

Each run writes one JSON file per suite and backend:

```
results/{suite}_{backend}.json
```

- `suite` ∈ `{so, mo}`
- `backend` ∈ `{torch-cpu, torch-cuda, torch-rocm, evox-numpy,
  evox-iree-llvm-cpu, evox-iree-llvm-cpu_mi300x, evox-iree-cuda, evox-iree-rocm,
  evox-xla-cuda, evox-xla-rocm}`

A `--out <path>` override is also supported (used by smoke tests, which
never touch this directory).

Only the `evox-*` / `evox-*-mi300x` names are producible today — the `torch-*`
ids above are listed only to document the historical files.

### Host distinction (A6000/CUDA vs MI300X/ROCm)

The committed `{suite}_{backend}.json` files for `evox-numpy`,
`evox-iree-llvm-cpu` (no suffix), `torch-cpu`, `torch-cuda`, `evox-iree-cuda`
and `evox-xla-cuda` were measured on the A6000/CUDA host and still back the
CUDA sections of `../BENCHMARK_RESULTS.md` — never overwrite them with
MI300X numbers. Measurements taken on the MI300X/ROCm host use:

- the natural `*-rocm` names for the three ROCm ids —
  `{suite}_torch-rocm.json`, `{suite}_evox-iree-rocm.json`,
  `{suite}_evox-xla-rocm.json`;
- a `_mi300x` suffix when re-running the CPU column on this host —
  `{suite}_evox-iree-llvm-cpu_mi300x.json` (pass `--out`; the runner's default
  name would overwrite the A6000 file).

`{suite}_torch-rocm.json` holds torch-on-HIP numbers: torch's device string
is `"cuda"` — there is no `"rocm"` torch device.

Committed so far: all 12 CUDA-host files (36 SO + 12 MO records each, 288
total).
The post-fix re-run (2026-09-04) refreshed every CMA-ES SO cell on all six
backends and the MO suite on the four evox backends after two upstream fixes
(CMA-ES `c_c` Hansen-canonical formula in torch AND evox; DTLZ
`cumprod`/`flip` replaced with exportable ops). The large-scale CMA-ES cells
now execute everywhere (no more cusolver/`ValueError`/silent-NaN records).
Pareto fronts are extracted per-generation on both torch and evox paths
(`_torch_pf_fitness` / `_evox_pf_fitness` in `bench_mo.py`) — `EvalMonitor.get_pf_fitness`'s
O(n²) whole-history domination matrix OOM-kills the process at 1000-pop ×
100-gen scale; the two methods are mathematically identical (verified
point-for-point at small scale). The 2026-09-05 re-run refreshed all 9
CMA-ES SO cells on all six backends after the `p_c` quirk fix (the evox side,
scalar-dot-product replication); the CMAES/Ackley/100x10 plateau stall is
resolved on all four evox backends (see BENCHMARK_RESULTS.md). The
2026-09-05 NSGA3 re-run refreshed the 4 NSGA3 MO cells on all four evox
backends after the `matrix_rank`/`solve` → eigh export fix: evox-numpy
(bit-identical hv/igd, refreshed timings) and evox-xla-cuda (4/4 cells now
run) succeed; both iree backends now pass export and crash in the known
iree-compile segfault (see below). The MI300X/ROCm round on top of that added
8 files — `{so,mo}_torch-rocm.json`, `{so,mo}_evox-xla-rocm.json`,
`{so,mo}_evox-iree-rocm.json`, `{so,mo}_evox-iree-llvm-cpu_mi300x.json` — plus
`rocm_prefix_baseline.json`; see "ROCm re-run (this host)" below. The 12
CUDA-host files and the schema below are unchanged by that round.

## JSON schema

The file is a JSON array with one object per case:

| field | type | meaning |
|---|---|---|
| `case_id` | str | e.g. `PSO/Sphere/100x10`, `NSGA2/DTLZ2/100x3x10` |
| `backend` | str | backend under test |
| `algo` / `problem` | str | algorithm / problem names |
| `pop_size`, `dim`, `n_obj` | int | case geometry (`n_obj` = 1 for SO) |
| `gens` | int | timed generations actually run |
| `seed` | int | RNG seed (42; torch `manual_seed`, evox `init(seed=...)`) |
| `compile_time_s` | float | torch: 0.0 (eager, `note="eager"`); evox: wall time of `wf.init(seed=42)` (graph build + backend compile) |
| `warmup_steps` | int | always 2 (steps run before the timed window) |
| `run_time_s` | float | wall time of the timed `gens` steps (torch-cuda syncs before/after) |
| `ms_per_step` | float | `1000 * run_time_s / gens` |
| `metric` | object | SO: `{"best_fitness": f}` (monitor elite, minimized). MO: `{"hv": f, "igd": f, "n_pf_points": n}` |
| `parity` | object | vs the `results/{suite}_torch-cpu.json` baseline; `{vs, rel_err, ok}` (+`rel_err_hv`/`rel_err_igd` for MO). Absent for the torch-cpu baseline itself, or when no baseline file exists yet. `ok = rel_err <= 0.10` |
| `error` | str | `"<Type>: <msg>"` when the case failed (the runner continues) |
| `note` | str | `"eager"` / `"compiled (<evox backend>)"`; MOEAD cases note that the effective population is the Das-Dennis count; capped evox-numpy SO records note the gens cap; failed iree MO cells carry per-cell notes naming the precise failure (`iree-compile segfault (SIGSEGV, error -11)` / runtime `ref is null`; post-fix NSGA3 iree cells note that export now passes and the compiler segfaults). Record-level notes may also carry slow-path/lowering annotations (free text; does not affect parity verdicts) |

## `rocm_prefix_baseline.json` — pre-fix ROCm baseline

Not a runner output and not a per-case array: it is the **pre-fix** ROCm
matrix reconstructed from the previous round's scratch table dumps, because
no `results/*_rocm.json` file was committed for that round. It exists so
every pre-fix ROCm number quoted in `../BENCHMARK_RESULTS.md` traces back to
a file in this directory.

Schema:

| key | meaning |
|---|---|
| `provenance` | block with `description`, `source_files` (the scratch markdown dumps), `source_note`, `date`, `host` (AMD Instinct MI300X, one VF, ROCm 7.1, torch 2.13.0+rocm7.1, Python 3.14), `evox_build` (pre-fix evox: before the batched-eigh default, two-stage split reduction, XLA per-call hot path, emitter correctness fixes and the pipelined monitor readback), `case_config` (seed 42, 100 gens — SO `evox-iree-llvm-cpu` CMA-ES 10000x100 cells capped at 22 gens — timed window excludes 2 warmup steps), `units` (`ms_per_step`), `reconstruction` (`err` = error record, `@Ng` = gens cap) |
| `so` | `{case_id: {backend: {ms_per_step, best_fitness}}}` |
| `mo` | `{case_id: {backend: {ms_per_step, hv, igd}}}`; error records carry `ms_per_step: null, error: true, hv: null, igd: null` |

Backend keys inside `so`/`mo`: `torch-rocm`, `evox-xla-rocm`, `evox-iree-rocm`,
`evox-numpy`, `evox-iree-llvm-cpu`. There are no `parity` blocks — it predates
the per-case schema. Its values are pre-fix scratch measurements (reference
only, never current numbers), and its `evox-numpy` MI300X columns are the
numbers REUSED in the current tables because the numpy backend was not
re-run on this host.

## Config (shared by all backends)

- SO: PSO/DE use `lb=-100, ub=100` per dim (default hyperparameters);
  CMA-ES `mean_init=50.0·1, sigma=25.0`; OpenES `center_init=50.0·1,
  learning_rate=0.01, noise_stdev=5.0`. Problems: Sphere/Rastrigin/Ackley.
  Scales (pop, dim): (100,10), (1000,50), (10000,100).
- MO: `lb=0, ub=1` per dim (DTLZ domain). Problems: DTLZ1/DTLZ2.
  Scales (pop, n_obj, dim): (100,3,10), (1000,3,30).
- 100 generations, seed 42 (override with `--gens-override`).

## Metrics (identical numpy code for every backend — see `bench_common.py`)

- `hv`: bounding-cube Monte Carlo, 100_000 samples, seed 0 — same math as
  `src/evox/metrics/hv.py`. Reference point `[2.0] * n_obj` for both DTLZ1
  and DTLZ2 (dominates every PF point of both; documented in the source).
- `igd`: mean over reference-front points of the min L1 distance (p=1.0),
  same math as `src/evox/metrics/igd.py`. Reference fronts (fixed seed 0):
  DTLZ1 → 1000 points on the simplex `sum(f)=0.5`; DTLZ2 → 1000 points on
  the positive orthant of the unit sphere.

## Timing semantics

- torch-cuda: `torch.cuda.synchronize()` before and after the timed window,
  so queued kernel execution is included.
- evox xla/iree adapters stage inputs from host buffers and copy outputs
  back to host on every `evoxc.run`, so per-step wall time includes the
  device↔host transfer cost inherent to the current adapter design.
- evox-xla-cuda runs with the adapter's cpu device label (`device=None`):
  the current xla adapter rejects non-cpu devices at load ("the xla adapter
  supports only CPU devices"); the CUDA PJRT plugin has no CPU platform, so
  execution still lands on the visible GPU (GPU selected purely via
  `CUDA_VISIBLE_DEVICES`).

## GPU environment recipe (applied by `run.py`/`--gpu` before any import)

`CUDA_VISIBLE_DEVICES=<id>`,
`LD_LIBRARY_PATH=/mnt/local-ssd/bchuang/cudnn-xla/lib` (prepended),
`XLA_FLAGS=--xla_gpu_cuda_data_dir=/home/bchuang/xla_cuda_data`,
`EVOXC_PJRT_PLUGIN=<venv>/jax_plugins/xla_cuda12/xla_cuda_plugin.so`.

For `evox-xla-cuda` the harness additionally prepends
`/mnt/local-ssd/bchuang/cudnn-xla/lib/libcudnn.so.9` to `LD_PRELOAD` and
**re-execs the process** (marker-guarded, idempotent), because:
1. the xla PJRT plugin was compiled against cuDNN 9.8 but carries a
   DT_RPATH pointing at the venv's pip cuDNN 9.1.0 — DT_RPATH beats
   `LD_LIBRARY_PATH` for the plugin's own `dlopen("libcudnn.so.9")`, so the
   prescribed `LD_LIBRARY_PATH` alone cannot fix the version (every PJRT
   compile then crashes with `RET_CHECK failure (gpu_compiler.cc:2798)
   dnn_support != nullptr`);
2. the dynamic loader reads `LD_PRELOAD` only at process start, so a plain
   in-process `os.environ` mutation is too late — hence the re-exec.

## Parity caveat

torch (MT19937) and evox (keyed PRNG) draw different RNG streams, so runs are
never trajectory-identical — `parity.ok` compares converged fitness/hv/igd
within a 10% relative tolerance, matching `unit_test/evox/algorithms/parity/`.
At very few generations (e.g. the 3-gen smoke runs) the relative error is
naturally large (`ok=false`); the tolerance is meaningful for full 100-gen
runs. The relative-error metric also false-alarms when both sides converge
to ~0 (e.g. `PSO/Sphere/100x10`, `CMAES/Sphere/100x10`).

## Committed post-fix re-run (2026-09-04)

After the two upstream fixes (CMA-ES `c_c` Hansen-canonical formula in torch
AND evox; DTLZ `cumprod`/`flip` → exportable gather/reduce_prod), the
following cells were re-run and merged into the committed JSONs (see
`retag_parity.py` for the merge + verdict conventions):

- **SO CMA-ES: all 9 cells on all 6 backends** (100 gens; evox-numpy keeps
  its interpreter caps: 50 gens at 1000x50, 10 gens at 10000x100). All
  previous error/NaN records are replaced with real metrics; the torch-cpu
  baseline was re-run first so parity verdicts reference it.
- **MO: all 12 cells on evox-numpy (now full 100 gens — the O(n²)
  whole-history ranking was replaced by per-generation extraction, removing
  the previous 20-gen cap), evox-iree-llvm-cpu, evox-iree-cuda,
  evox-xla-cuda.** xla-cuda runs NSGA2+MOEAD end-to-end (8/12 cells);
  the remaining cells fail with the NEW blockers below.
- **SO CMA-ES 2026-09-05 re-run:** all 9 CMA-ES SO cells re-run on all 6
  backends after the `p_c` quirk fix; 0 error records; the
  CMAES/Ackley/100x10 stall resolved (new evox bests ≈ 0.00055–0.0037);
  parity verdicts re-derived with `retag_parity.py`; torch-cpu baseline
  re-run first (fitness bit-identical, timing refreshed).
- **MO NSGA3 re-run (2026-09-05):** all 4 NSGA3 cells re-run on evox-numpy,
  evox-xla-cuda, evox-iree-llvm-cpu, evox-iree-cuda after the matrix_rank/solve
  → eigh fix. numpy: 4/4 ok (hv/igd bit-identical to pre-fix records — the
  replacement is numerically equivalent on the numpy backend; timings
  refreshed). xla-cuda: 4/4 ok (previously 4 export errors) — DTLZ2 parity
  ok (rel 0.009/0.071), DTLZ1 not-ok RNG scatter (rel 16.636/0.528).
  iree-llvm-cpu / iree-cuda: export now passes; all 4 cells crash in the
  iree-compile segfault — verified NOT to be an NSGA3 regression (identical
  to NSGA2/MOEAD; do not attempt to fix iree). Merged with
  retag_parity.py --partial + --set-note per the committed conventions.

Known issues recorded in the JSONs after the re-run:

1. **iree-compile segfault on all MO programs — 23 cells** (iree-cuda 12,
   iree-llvm-cpu 11): `Error code: -11`, stack dump inside
   `libIREECompiler.so` (deep recursion). The stablehlo v1 export succeeds;
   the compiler itself crashes. Not retryable. NSGA3 reached this crash only
   after the `matrix_rank`/`solve` → eigh fix (2ae931e4); the unmodified
   NSGA2/MOEAD tell triggers it identically — an upstream iree while-loop
   compiler bug.
2. **MOEAD/DTLZ2/100x3x10 on iree-llvm-cpu** passes export AND compile but
   fails at runtime: `ref is null ... hal.buffer_view.create`
   (INVALID_ARGUMENT).
3. Large-scale CMA-ES Ackley cells (1000x50/10000x100) remain unconverged on
   torch-cuda (≈20.6/20.4) and scatter on the evox compiled backends (rel
   0.44–1.89) — RNG-stream variance on unconverged runs; the 100x10 stall
   itself is resolved by the `p_c` quirk fix.

## ROCm re-run (this host)

Host/env: AMD Instinct MI300X (one VF), ROCm 7.1, venv
`/tmp/genesis/task_3858dfe003526650/etlbench-venv/bin/python` (Python 3.14,
torch 2.13.0+rocm7.1), evoxc 0.1.0, `evox` imported from the repo `src/`.
Runs were serial and foreground, seed 42, 100 gens (caps noted below), the
timed window excluding 2 warmup steps, with case tables/geometry identical to
the committed CUDA runs.

- **Files written:** `{so,mo}_torch-rocm.json`, `{so,mo}_evox-xla-rocm.json`,
  `{so,mo}_evox-iree-rocm.json`, `{so,mo}_evox-iree-llvm-cpu_mi300x.json`; the
  pre-fix reference is `rocm_prefix_baseline.json`.
- **SO:** `torch-rocm` 36 records of which the 9 CMA-ES cells are `err`;
  `evox-xla-rocm`, `evox-iree-rocm` and `evox-iree-llvm-cpu_mi300x` are 36/36 ok
  (llvm-cpu CMA-ES 10000x100 at the full 100 gens — the pre-fix 22-gen cap is
  gone).
- **MO:** `torch-rocm` 12/12 and `evox-xla-rocm` 12/12 ok;
  `evox-iree-rocm` 12/12 `err` (iree-compile SIGSEGV, `Error code: -11`, on
  the `rocm` target, deterministic across 4 in-process + 4 fresh-process
  attempts per case); `evox-iree-llvm-cpu_mi300x` 3/12 run
  (`MOEAD/DTLZ1/100x3x10`, `MOEAD/DTLZ1/1000x3x30`,
  `MOEAD/DTLZ2/1000x3x30`) and 9/12 `err` — MO compile on llvm-cpu is
  nondeterministic, so those three needed process-level retries (the
  documented `EVOXC_COMPILE_ATTEMPTS` recipe).
- **Reused, not re-measured: `evox-numpy`.** The numpy backend is unaffected
  by the evoxc device-path fixes, so its MI300X numbers come from
  `rocm_prefix_baseline.json` and must be labelled as reused.
- **Round caveats:** torch CMA-ES is unavailable on this host
  (`UncapturedHigherOrderOpError ... torch.cond`, a torch 2.13
  incompatibility that also reproduces on torch-cpu — the torch reference for
  CMA-ES does not exist, do not chase it); `torch-rocm` MO timings vary
  between full sweeps (−0.6% to +7.1% per cell) and NSGA3 carries a larger
  ~±20% run-to-run spread (186 ms/step for `NSGA3/DTLZ2/100x3x10` in a
  two-case spot run vs 254–272 in the full sweeps) and is not point-for-point
  reproducible across GPU runs; a
  `torch-rocm` process can abort in the HIP teardown path
  (`c10::cuda::SetDevice` inside a `TensorImpl` destructor) AFTER the JSON is
  written — stdout/the JSON is the source of truth, it is not a failed run.
- **ROCm env recipe:** `HIP_VISIBLE_DEVICES` + `ROCR_VISIBLE_DEVICES`; the xla
  id additionally sets `EVOXC_PJRT_PLUGIN` to the venv's
  `jax_plugins/xla_rocm7/xla_rocm_plugin.so` and appends
  `XLA_FLAGS=--xla_gpu_enable_command_buffer=` (HIP command buffers SIGSEGV
  inside `PJRT_Executable_Execute`). The ROCm device is single-tenant — never
  run two ROCm jobs at once.
