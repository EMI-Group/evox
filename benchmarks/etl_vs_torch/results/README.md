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

Committed so far: `so_torch-cuda.json` / `mo_torch-cuda.json` (full matrix,
GPU 0, 100 gens, seed 42). The 6 large-scale CMA-ES torch-cuda cases
(1000x50 and 10000x100) are error records — see the `note` field and
`benchmarks/etl_vs_torch/CONTEXT.md` for the root cause (evox CMA-ES `c_c`
formula bug → NaN covariance; cusolver raises, CPU LAPACK silently
propagates the NaN, so the torch-cpu baseline is wrong there too).
Torch-side MO Pareto fronts are extracted per-generation (not via
`EvalMonitor.get_pf_fitness`, whose O(n²) domination matrix gets the process
OOM-killed at 1000-pop scale); the two methods are mathematically identical
(verified point-for-point at small scale).

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
| `note` | str | `"eager"` / `"compiled (<etl backend>)"`; MOEAD cases note that the effective population is the Das-Dennis count; iree MO failures carry `"documented: iree-cuda while-loop shape issue"`; large-scale torch CMA-ES failures carry `"documented: evox CMA-ES c_c formula bug (torch.sqrt of a negative value -> NaN covariance) at large mu_eff/dim (pop>=1000, dim>=50); cusolver raises, CPU LAPACK silently propagates the NaN"` |

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

## Committed full-matrix run (etl-xla-cuda, GPU 3, 2026-09-04)

`so_etl-xla-cuda.json` (36 records) + `mo_etl-xla-cuda.json` (12 records) are
the full 100-gen matrix for the etl-xla-cuda backend; `so_torch-cpu.json` is
the parity baseline generated alongside (run artifacts are committed for
this run by task directive — see `results/` history for details). Known
issues recorded in the JSONs:

1. **MO: 12/12 `BackendError`** — the stablehlo v1 exporter (used by the xla
   adapter) defers the `cumprod` (and `flip`) ops used by
   `src/evox_etl/problems/numerical/dtlz.py`; every MO case fails at compile
   time. The numpy backend runs the same cases fine, and both etl checkouts
   (`/mnt/local-ssd/bchuang/etl`, `/mnt/local-ssd/bchuang/etl-xla-fixed`)
   defer identically. Unblocking requires decomposing those ops in the etl
   exporter or rewriting dtlz.py's evaluate formulas with v1-safe ops
   (static trace-time loops over `multiply`/`concatenate` — `m` is static),
   then re-running the MO groups.
2. **CMA-ES pop ≫ dim: 6/6 `ValueError: math domain error`** — the classic
   Hansen parametrization gives `c_c > 2` at (pop=1000, dim=50) and
   (pop=10000, dim=100), so `sqrt(c_c·(2−c_c)·mu_eff)` goes negative and the
   etl port raises at trace time. The torch implementation uses the same
   formulas and silently corrupts (NaN) instead — same root cause, both
   sides. Deterministic; not retryable.
3. **`CMAES/Ackley/100x10` parity fails (rel ≈ 6584)** — torch-cpu converges
   (best ≈ 0.003, all 7 seeds tested); the etl CMA-ES port stalls on
   Ackley's flat plateau (best ≈ 20.35–20.6 on etl-numpy AND etl-xla-cuda;
   sigma random-walks ~20–31, mean never descends; etl-numpy converges only
   1/7 seeds). Not xla-specific — the port's dynamics diverge from torch's
   on near-flat landscapes and need attention in `src/evox_etl`.
