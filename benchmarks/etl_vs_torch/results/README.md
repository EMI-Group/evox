# results/ — evox (torch) vs evox_etl benchmark outputs

## Naming convention

Each run writes one JSON file per suite and backend:

```
results/{suite}_{backend}.json
```

- `suite` ∈ `{so, mo}`
- `backend` ∈ `{torch-cpu, torch-cuda, etl-numpy, etl-iree-llvm-cpu,
  etl-iree-cuda, etl-xla-cuda}`

A `--out <path>` override is also supported (used by smoke tests, which
never touch this directory).

Committed so far: all 12 files (36 SO + 12 MO records each, 288 total).
The post-fix re-run (2026-09-04) refreshed every CMA-ES SO cell on all six
backends and the MO suite on the four etl backends after two upstream fixes
(CMA-ES `c_c` Hansen-canonical formula in torch AND etl; DTLZ
`cumprod`/`flip` replaced with exportable ops). The large-scale CMA-ES cells
now execute everywhere (no more cusolver/`ValueError`/silent-NaN records).
Pareto fronts are extracted per-generation on both torch and etl paths
(`_torch_pf_fitness` / `_etl_pf_fitness` in `bench_mo.py`) — `EvalMonitor.get_pf_fitness`'s
O(n²) whole-history domination matrix OOM-kills the process at 1000-pop ×
100-gen scale; the two methods are mathematically identical (verified
point-for-point at small scale). The 2026-09-05 re-run refreshed all 9
CMA-ES SO cells on all six backends after the `p_c` quirk fix (etl-side
scalar-dot-product replication); the CMAES/Ackley/100x10 plateau stall is
resolved on all four etl backends (see BENCHMARK_RESULTS.md). The
2026-09-05 NSGA3 re-run refreshed the 4 NSGA3 MO cells on all four etl
backends after the `matrix_rank`/`solve` → eigh export fix: etl-numpy
(bit-identical hv/igd, refreshed timings) and etl-xla-cuda (4/4 cells now
run) succeed; both iree backends now pass export and crash in the known
iree-compile segfault (see below).

## JSON schema

The file is a JSON array with one object per case:

| field | type | meaning |
|---|---|---|
| `case_id` | str | e.g. `PSO/Sphere/100x10`, `NSGA2/DTLZ2/100x3x10` |
| `backend` | str | backend under test |
| `algo` / `problem` | str | algorithm / problem names |
| `pop_size`, `dim`, `n_obj` | int | case geometry (`n_obj` = 1 for SO) |
| `gens` | int | timed generations actually run |
| `seed` | int | RNG seed (42; torch `manual_seed`, etl `init(seed=...)`) |
| `compile_time_s` | float | torch: 0.0 (eager, `note="eager"`); etl: wall time of `wf.init(seed=42)` (graph build + backend compile) |
| `warmup_steps` | int | always 2 (steps run before the timed window) |
| `run_time_s` | float | wall time of the timed `gens` steps (torch-cuda syncs before/after) |
| `ms_per_step` | float | `1000 * run_time_s / gens` |
| `metric` | object | SO: `{"best_fitness": f}` (monitor elite, minimized). MO: `{"hv": f, "igd": f, "n_pf_points": n}` |
| `parity` | object | vs the `results/{suite}_torch-cpu.json` baseline; `{vs, rel_err, ok}` (+`rel_err_hv`/`rel_err_igd` for MO). Absent for the torch-cpu baseline itself, or when no baseline file exists yet. `ok = rel_err <= 0.10` |
| `error` | str | `"<Type>: <msg>"` when the case failed (the runner continues) |
| `note` | str | `"eager"` / `"compiled (<etl backend>)"`; MOEAD cases note that the effective population is the Das-Dennis count; capped etl-numpy SO records note the gens cap; failed iree MO cells carry per-cell notes naming the precise failure (`iree-compile segfault (SIGSEGV, error -11)` / runtime `ref is null`; post-fix NSGA3 iree cells note that export now passes and the compiler segfaults). Record-level notes may also carry slow-path/lowering annotations (free text; does not affect parity verdicts) |

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
- etl xla/iree adapters stage inputs from host buffers and copy outputs
  back to host on every `etl.run`, so per-step wall time includes the
  device↔host transfer cost inherent to the current adapter design.
- etl-xla-cuda runs with the adapter's cpu device label (`device=None`):
  the current xla adapter rejects non-cpu devices at load ("the xla adapter
  supports only CPU devices"); the CUDA PJRT plugin has no CPU platform, so
  execution still lands on the visible GPU (GPU selected purely via
  `CUDA_VISIBLE_DEVICES`).

## GPU environment recipe (applied by `run.py`/`--gpu` before any import)

`CUDA_VISIBLE_DEVICES=<id>`,
`LD_LIBRARY_PATH=/mnt/local-ssd/bchuang/cudnn-xla/lib` (prepended),
`XLA_FLAGS=--xla_gpu_cuda_data_dir=/home/bchuang/xla_cuda_data`,
`ETL_PJRT_PLUGIN=<venv>/jax_plugins/xla_cuda12/xla_cuda_plugin.so`.

For `etl-xla-cuda` the harness additionally prepends
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

torch (MT19937) and etl (keyed PRNG) draw different RNG streams, so runs are
never trajectory-identical — `parity.ok` compares converged fitness/hv/igd
within a 10% relative tolerance, matching `unit_test/etl/algorithms/parity/`.
At very few generations (e.g. the 3-gen smoke runs) the relative error is
naturally large (`ok=false`); the tolerance is meaningful for full 100-gen
runs. The relative-error metric also false-alarms when both sides converge
to ~0 (e.g. `PSO/Sphere/100x10`, `CMAES/Sphere/100x10`).

## Committed post-fix re-run (2026-09-04)

After the two upstream fixes (CMA-ES `c_c` Hansen-canonical formula in torch
AND etl; DTLZ `cumprod`/`flip` → exportable gather/reduce_prod), the
following cells were re-run and merged into the committed JSONs (see
`retag_parity.py` for the merge + verdict conventions):

- **SO CMA-ES: all 9 cells on all 6 backends** (100 gens; etl-numpy keeps
  its interpreter caps: 50 gens at 1000x50, 10 gens at 10000x100). All
  previous error/NaN records are replaced with real metrics; the torch-cpu
  baseline was re-run first so parity verdicts reference it.
- **MO: all 12 cells on etl-numpy (now full 100 gens — the O(n²)
  whole-history ranking was replaced by per-generation extraction, removing
  the previous 20-gen cap), etl-iree-llvm-cpu, etl-iree-cuda,
  etl-xla-cuda.** xla-cuda runs NSGA2+MOEAD end-to-end (8/12 cells);
  the remaining cells fail with the NEW blockers below.
- **SO CMA-ES 2026-09-05 re-run:** all 9 CMA-ES SO cells re-run on all 6
  backends after the `p_c` quirk fix; 0 error records; the
  CMAES/Ackley/100x10 stall resolved (new etl bests ≈ 0.00055–0.0037);
  parity verdicts re-derived with `retag_parity.py`; torch-cpu baseline
  re-run first (fitness bit-identical, timing refreshed).
- **MO NSGA3 re-run (2026-09-05):** all 4 NSGA3 cells re-run on etl-numpy,
  etl-xla-cuda, etl-iree-llvm-cpu, etl-iree-cuda after the matrix_rank/solve
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
   torch-cuda (≈20.6/20.4) and scatter on the etl compiled backends (rel
   0.44–1.89) — RNG-stream variance on unconverged runs; the 100x10 stall
   itself is resolved by the `p_c` quirk fix.
