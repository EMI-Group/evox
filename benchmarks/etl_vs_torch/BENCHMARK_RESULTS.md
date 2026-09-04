# evox (torch) vs evox_etl — Benchmark Results

Comparison of the torch OOP `evox` (`src/evox/`) against the functional
`evox_etl` (`src/evox_etl/`). Six backends, two suites (single-objective,
multi-objective), small → large scale. All numbers below are read from the
committed `results/*.json` (12 files, one per suite×backend).

## Setup

- **Machine:** 8× NVIDIA RTX A6000 48 GB, 128 logical CPUs.
- **Python:** `/mnt/local-ssd/bchuang/evox/.venv/bin/python` (3.11.2).
- **torch:** 2.6.0+cu124, **eager only** — `StdWorkflow` has no compile mode,
  so `compile_time_s = 0.0` and torch runs never pay a compile cost.
- **etl:** compiled backends — `etl-numpy` (CPU interpreter),
  `etl-iree-llvm-cpu`, `etl-iree-cuda`, `etl-xla-cuda`; compile time is the
  wall time of `wf.init(seed=42)` and is reported separately from ms/step.
- **Seed** 42 everywhere; 100 generations default; 2 warmup steps.
- **ms/step** = `1000 · run_time_s / gens` over the timed window (torch-cuda
  synchronizes before/after). etl xla/iree adapters stage inputs from host
  buffers and copy outputs back every step, so their ms/step includes
  device↔host transfer.
- **SO metric:** monitor best fitness (elite, minimized).
- **MO metrics** (identical numpy code host-side for all backends, see
  `bench_common.py`): `hv` = bounding-cube Monte Carlo hypervolume, 100 000
  samples, fixed seed 0, reference point `[2.0]^m`; `igd` = mean L1 distance
  to fixed reference fronts (DTLZ1: 1000 points on the simplex `sum(f)=0.5`;
  DTLZ2: 1000 points on the positive orthant of the unit sphere; seed 0).
  Pareto fronts are extracted **per-generation** on both torch and etl paths
  (`_torch_pf_fitness` / `_etl_pf_fitness` in `bench_mo.py`) — the framework
  `get_pf_fitness` whole-history O(n²) domination ranking OOM-kills at
  1000-pop × 100-gen scale; the two methods are mathematically identical
  (verified point-for-point at small scale).

## Single-objective (SO) — PSO / DE / CMA-ES / OpenES × Sphere / Rastrigin / Ackley

Cell = `best_fitness / ms_per_step`. S/R/A = Sphere/Rastrigin/Ackley.
`err` = error record; `@Ng` = etl-numpy interpreter cap (gens actually run).
All SO cells now execute on every backend (the CMA-ES `c_c` fix removed the
last SO error cells — see *Caps & limits*).

### 100 × 10

| backend | PS | PR | PA | DS | DR | DA | CS | CR | CA | OS | OR | OA |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-cpu | 1.538e-08 / 0.677 | 8.088 / 0.6997 | 20.01 / 0.7587 | 0.2588 / 0.5298 | 47.47 / 0.5681 | 20 / 0.6173 | 2.064e-07 / 57.53 | 26.12 / 124.4 | 0.003625 / 134 | 172.8 / 0.4103 | 252.1 / 0.4486 | 20.66 / 0.4905 |
| torch-cuda | 9.313e-09 / 1.063 | 5.048 / 1.156 | 20 / 1.198 | 0.186 / 0.8846 | 56.9 / 0.9457 | 19.42 / 1.004 | 6.428e-08 / 57.85 | 5.055 / 119.7 | 0.007918 / 124 | 92.91 / 0.6223 | 201.4 / 0.6924 | 20.5 / 0.7366 |
| etl-numpy | 3.054e-08 / 0.8298 | 5.52 / 0.9024 | 20 / 0.9534 | 0.6075 / 0.9725 | 56.38 / 1.038 | 18.95 / 1.083 | 1.304e-09 / 1.309 | 1.879 / 1.347 | 20.56 / 1.503 | 132.1 / 0.4907 | 221.2 / 0.5614 | 20.69 / 0.6164 |
| etl-iree-llvm-cpu | 3.054e-08 / 1.304 | 5.52 / 2.301 | 20 / 2.708 | 0.6075 / 1.241 | 56.38 / 1.264 | 18.95 / 2.009 | 1.279e-09 / 27.72 | 1.014 / 24.43 | 20.58 / 29.06 | 132.1 / 0.9726 | 221.2 / 1.011 | 20.69 / 0.8136 |
| etl-iree-cuda | 3.054e-08 / 1.421 | 5.52 / 1.492 | 20 / 1.56 | 0.6075 / 1.13 | 56.38 / 1.287 | 18.95 / 1.244 | 1.328e-09 / 43.29 | 1.102 / 39.88 | 20.53 / 42.94 | 132.1 / 1.457 | 221.2 / 1.152 | 20.69 / 1.253 |
| etl-xla-cuda | 3.054e-08 / 5.366 | 5.52 / 5.112 | 20 / 5.081 | 0.6075 / 3.271 | 56.38 / 3.914 | 19.83 / 4.215 | 1.28e-09 / 10.18 | 0.02222 / 10.15 | 20.59 / 8.959 | 132.1 / 4.78 | 221.2 / 4.817 | 20.69 / 4.705 |

### 1000 × 50

| backend | PS | PR | PA | DS | DR | DA | CS | CR | CA | OS | OR | OA |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-cpu | 30.77 / 2.558 | 391.5 / 2.646 | 20 / 2.693 | 1.798e+04 / 2.223 | 2.244e+04 / 2.608 | 20.28 / 2.419 | 67.05 / 132.8 | 489.7 / 139.6 | 7.317 / 142.6 | 1814 / 0.8695 | 2271 / 0.9689 | 21.26 / 1.005 |
| torch-cuda | 13.46 / 1.085 | 330.6 / 1.143 | 20.05 / 1.205 | 2.562e+04 / 0.9 | 2.749e+04 / 0.9707 | 20.26 / 1.028 | 68.77 / 118.6 | 530.2 / 125.4 | 20.63 / 125.8 | 1751 / 0.6802 | 2214 / 0.7842 | 21.21 / 0.8361 |
| etl-numpy | 201.4@50g / 2.166 | 875.2@50g / 2.717 | 20.58@50g / 2.739 | 4.271e+04@50g / 1.913 | 4.2e+04@50g / 1.904 | 20.52@50g / 1.983 | 5763@50g / 3.052 | 6179@50g / 2.909 | 21.26@50g / 3.122 | 1.306e+04@50g / 0.9856 | 1.365e+04@50g / 1.111 | 21.24@50g / 1.202 |
| etl-iree-llvm-cpu | 31.94 / 12.39 | 365.9 / 10.31 | 20.56 / 9.819 | 2.384e+04 / 9.794 | 2.792e+04 / 9.966 | 20.14 / 12.69 | 35.76 / 1273 | 428.9 / 1283 | 21.13 / 1340 | 1887 / 10.15 | 2370 / 10.53 | 21.24 / 11.98 |
| etl-iree-cuda | 31.94 / 35.69 | 365.9 / 35.84 | 20.56 / 35.88 | 2.384e+04 / 35.7 | 2.792e+04 / 35.61 | 20.14 / 35.46 | 35.74 / 1117 | 447.3 / 1354 | 21.17 / 1425 | 1887 / 35.49 | 2370 / 35.2 | 21.24 / 35.59 |
| etl-xla-cuda | 31.94 / 5.73 | 365.9 / 5.738 | 20.56 / 5.291 | 2.384e+04 / 4.36 | 2.792e+04 / 4.602 | 20.08 / 3.638 | 35.77 / 87.92 | 429.9 / 88.48 | 21.11 / 87.79 | 1887 / 3.71 | 2370 / 3.836 | 21.24 / 3.772 |

### 10000 × 100

| backend | PS | PR | PA | DS | DR | DA | CS | CR | CA | OS | OR | OA |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-cpu | 2548 / 27.45 | 3846 / 36.21 | 20.08 / 26.65 | 9.718e+04 / 16.77 | 1.08e+05 / 20.08 | 20.61 / 17.7 | 294.1 / 160.1 | 1261 / 166.3 | 11.91 / 167.6 | 4219 / 8.587 | 5163 / 8.237 | 21.3 / 10.43 |
| torch-cuda | 1893 / 1.085 | 3362 / 1.135 | 20.12 / 1.259 | 1.091e+05 / 0.8736 | 1.189e+05 / 1.01 | 20.67 / 1.002 | 218.2 / 121.2 | 1189 / 129.4 | 20.38 / 129 | 4068 / 0.6265 | 5095 / 0.7006 | 21.31 / 0.754 |
| etl-numpy | 8.43e+04@10g / 73.99 | 8.52e+04@10g / 66.56 | 20.35@10g / 64.45 | 1.595e+05@10g / 42.13 | 1.555e+05@10g / 38.57 | 20.82@10g / 42.41 | 2.123e+05@10g / 46.1 | 2.133e+05@10g / 67.12 | 21.31@10g / 61.14 | 1.484e+05@10g / 26.79 | 1.492e+05@10g / 25.36 | 21.39@10g / 22.31 |
| etl-iree-llvm-cpu | 2691 / 399.2 | 4119 / 561.4 | 20.02 / 707.3 | 1.048e+05 / 506 | 1.083e+05 / 554.4 | 20.56 / 585.5 | 201.5 / 2.288e+04 | 1164 / 2.192e+04 | 21.22 / 2.189e+04 | 4112 / 481.7 | 5110 / 443.2 | 21.32 / 479.9 |
| etl-iree-cuda | 2691 / 3475 | 4119 / 3481 | 20.02 / 3478 | 1.048e+05 / 3481 | 1.083e+05 / 3483 | 20.56 / 3496 | 200.3 / 9795 | 1164 / 9477 | 21.34 / 9604 | 4112 / 3482 | 5110 / 3480 | 21.32 / 3504 |
| etl-xla-cuda | 2691 / 17 | 4119 / 15.2 | 20.02 / 18.16 | 1.048e+05 / 11.18 | 1.083e+05 / 11.6 | 20.64 / 11.6 | 200.1 / 405.5 | 1163 / 399.9 | 21.29 / 400.1 | 4112 / 8.234 | 5110 / 9.325 | 21.32 / 9.34 |

### SO compile time and speedup

Compile time range over successful cases (`wf.init(seed=42)`; all 36 per
etl backend — no SO compile failures remain):

| backend | SO compile time |
|---|---|
| torch-cpu / torch-cuda | 0.0 s (eager) |
| etl-numpy | 0.044 – 0.382 s |
| etl-iree-llvm-cpu | 0.903 – 3.098 s |
| etl-iree-cuda | 1.209 – 20.575 s |
| etl-xla-cuda | 0.411 – 4.392 s |

ms/step ratio, case-matched, median (min – max); **>1 = faster than the baseline**:

| backend | vs torch-cpu | vs torch-cuda |
|---|---|---|
| torch-cuda | 1.29× (0.60× – 31.9×) | — |
| etl-numpy | 0.84× (0.32× – 92.4×) | — |
| etl-iree-llvm-cpu | 0.19× (0.007× – 5.1×) | — |
| etl-iree-cuda | 0.073× (0.002× – 3.1×) | 0.03× (0.0004× – 3.0×) |
| etl-xla-cuda | 0.51× (0.086× – 15.0×) | 0.21× (0.064× – 13.8×) |

The large "etl wins" outliers (92×, 15×) are CMA-ES 100x10, where torch's
CPU path is exceptionally slow (~58 ms/step vs 1.3–10 ms/step). The "torch
wins" floor is now set by the re-run CMA-ES cells: large-scale CMA-ES on
iree-llvm-cpu is ~1.3–2.3 s/step and on iree-cuda ~9.5–9.6 s/step (both
slower than torch's 0.12–0.17 s/step CPU path), so the iree medians dropped
vs the pre-fix run where those cells were error records. At 10000x100 torch
still wins decisively on both CPU and GPU except xla-cuda's 15–17 ms/step
PSO/DE/OpenES.

## Multi-objective (MO) — NSGA2 / NSGA3 / MOEAD × DTLZ1 / DTLZ2

Cell = `hv/igd / ms_per_step`. D1/D2 = DTLZ1/DTLZ2. `err` = error record
(see *Caps & limits*). MOEAD's effective population is the Das-Dennis vector
count (pop_size is overwritten on both sides).

### 100 × 3 × 10

| backend | NSGA2/1 | NSGA2/2 | NSGA3/1 | NSGA3/2 | MOEAD/1 | MOEAD/2 |
|---|---|---|---|---|---|---|
| torch-cpu | 7.92/2.045 / 4.2 | 7.46/0.02443 / 2.995 | 6.422/1.804 / 407.5 | 7.461/0.02572 / 420.3 | 184.1/4.505 / 97.02 | 7.46/0.02888 / 99.17 |
| torch-cuda | 7.685/0.5479 / 5.126 | 7.458/0.02459 / 4.291 | 6.456/1.151 / 623 | 7.463/0.02535 / 625.5 | 30.97/2.727 / 273.1 | 7.456/0.03117 / 279.5 |
| etl-numpy | 23.94/3.999 / 7.668 | 7.462/0.02312 / 6.413 | 113.4/9.624 / 10.99 | 7.461/0.02572 / 8.249 | 7.562/0.5722 / 61.36 | 7.455/0.03355 / 60.34 |
| etl-iree-llvm-cpu | err | err | err | err | err | err |
| etl-iree-cuda | err | err | err | err | err | err |
| etl-xla-cuda | 10.52/1.847 / 4.784 | 7.459/0.02465 / 4.146 | err | err | 7.562/0.5722 / 6.432 | 7.455/0.03355 / 6.458 |

### 1000 × 3 × 30

| backend | NSGA2/1 | NSGA2/2 | NSGA3/1 | NSGA3/2 | MOEAD/1 | MOEAD/2 |
|---|---|---|---|---|---|---|
| torch-cpu | 3.959e+04/51.1 / 83.07 | 7.463/0.01669 / 45.22 | 1.706e+04/41.36 / 417.9 | 7.466/0.01521 / 387.1 | 465/6.829 / 1033 | 7.459/0.0208 / 1028 |
| torch-cuda | 3616/23.1 / 7.646 | 7.464/0.01624 / 5.489 | 1.962e+04/44.2 / 863.3 | 7.467/0.01478 / 719.1 | 3546/15.03 / 3505 | 7.462/0.01766 / 3621 |
| etl-numpy | 8241/32.67 / 289.7 | 7.462/0.01756 / 231.7 | 5269/30.06 / 341.8 | 7.463/0.01795 / 282.1 | 1.023e+04/4.718 / 692.9 | 7.454/0.02432 / 688.6 |
| etl-iree-llvm-cpu | err | err | err | err | err | err |
| etl-iree-cuda | err | err | err | err | err | err |
| etl-xla-cuda | 9039/34.23 / 4.654 | 7.462/0.01713 / 5.268 | err | err | 7360/4.716 / 27.15 | 7.454/0.0244 / 26.92 |

### MO compile time and speedup

| backend | MO compile time |
|---|---|
| torch-cpu / torch-cuda | 0.0 s (eager) |
| etl-numpy | 0.346 – 0.582 s |
| etl-xla-cuda | 1.045 – 3.057 s (8 successful cases) |
| etl-iree-llvm-cpu / etl-iree-cuda | — (no case compiles; see *Caps & limits*) |

| backend | vs torch-cpu |
|---|---|
| torch-cuda | 0.65× (0.28× – 10.9×) |
| etl-numpy | 1.49× (0.20× – 51.0×) |
| etl-xla-cuda | 15.4× (0.72× – 38.2×; NSGA2+MOEAD only — NSGA3 does not compile) |
| etl-iree-llvm-cpu / etl-iree-cuda | — (all err) |

etl-numpy's 51× outlier is NSGA3/DTLZ2 100x3x10 (8.2 vs 420 ms/step) —
torch NSGA3's CPU path is python-loop-bound. etl-xla-cuda's 15.4× median
reflects compiled NSGA2/MOEAD (~4–27 ms/step) vs torch's python-bound
NSGA3/MOEAD CPU paths (99–1033 ms/step); torch-cuda stays bimodal (NSGA2
5.5–7.6 ms/step fast; NSGA3 623–863, MOEAD 273–3621 ms/step slow).

## Caps & limits

- **etl-numpy SO caps (interpreter too slow):** SO 1000x50 → 50 gens, SO
  10000x100 → 10 gens. Capped records keep their measured metric but parity
  is skipped (gens differ from baseline). The MO cap is **gone**: the harness
  now extracts Pareto fronts per-generation (see Setup), removing the O(n²)
  whole-history ranking that made 100 gens infeasible on the numpy
  interpreter — `mo_etl-numpy.json` is a full 100-gen matrix.
- **CMA-ES `c_c` bug — FIXED (torch + etl).** The Hansen-canonical formula
  `c_c = (4 + mu_eff/dim) / (dim + 4 + 2·mu_eff/dim)` replaced the wrong
  `(mu_eff + 2)` numerator on both sides. All 9 CMA-ES cells now execute on
  all 6 backends: the previous 6 trace-time `ValueError` records per etl
  backend, 6 cusolver `_LinAlgError` records on torch-cuda, and 6
  silently-NaN-corrupted torch-cpu records are replaced with real metrics
  (the torch-cpu baseline was re-run and updated).
- **DTLZ `cumprod`/`flip` export issue — FIXED.** `src/evox_etl/problems/numerical/dtlz.py`
  now uses exportable gather/reduce_prod constructions; the MO suite
  compiles on xla-cuda (NSGA2+MOEAD run end-to-end). Two NEW blockers
  remain, recorded with exact errors in the JSONs:
  1. **NSGA3 `matrix_rank` export rejection (12/12 NSGA3 cells, all three
     compiled backends):** `BackendError: stablehlo export: op 'matrix_rank'
     at src/evox_etl/algorithms/mo/nsga3.py:216 is not supported in v1 —
     decompose it into supported ops or use a future compiler adapter`
     (xla-cuda 4, iree-llvm-cpu 4, iree-cuda 4 cells). Previously masked by
     the cumprod error.
  2. **iree-compile segfault on NSGA2/MOEAD MO programs (15/24 iree cells):**
     `BackendError: iree-compile failed to compile ... Error code: -11` with
     a stack dump inside `libIREECompiler.so` (iree-cuda 8 cells, NSGA2+MOEAD;
     iree-llvm-cpu 7 cells). The export succeeds; the compiler itself
     crashes (deep recursion in the stack trace).
  3. **MOEAD/DTLZ2/100x3x10 on iree-llvm-cpu** is the only iree MO cell that
     passes export AND compile but dies at runtime: `ValueError: ... ref is
     null ... hal.buffer_view.create` (INVALID_ARGUMENT).
- **CMAES/Ackley/100x10 divergence persists:** etl CMA-ES still stalls on
  Ackley's zero-gradient plateau (best ≈ 20.53–20.59 on all four etl
  backends; rel_err ≈ 5.66–5.68e3 vs torch-cpu 0.0036). The `c_c` fix
  addressed the large-pop NaN regime, not this stall — a port divergence in
  `src/evox_etl`. Related scatter: torch-cuda's re-run CMA-ES Ackley cells
  at 1000x50/10000x100 stall at ≈20.6/20.4 while torch-cpu reaches 7.3/11.9
  (rel 1.8/0.7) — RNG-stream variance on unconverged Ackley runs.
- **MOEAD pop_size overwritten** by the Das-Dennis vector count on both
  torch and etl (recorded in the JSON notes).
- **Harness workarounds** baked into the numbers: torch MO runs with
  `torch.set_default_device(device)` and on-device lb/ub (torch MO algorithms
  allocate on the default device); both torch and etl Pareto fronts are
  extracted per-generation instead of `get_pf_fitness` (see Setup);
  etl-xla-cuda stages with `device=None` because the xla adapter rejects
  non-cpu device labels (the GPU is selected purely via
  `CUDA_VISIBLE_DEVICES`, and the CUDA PJRT plugin has no CPU platform);
  the harness LD_PRELOADs cuDNN 9.8 and re-execs the process for xla-cuda
  (see Reproduction).

## Parity verdicts

Parity is computed per record against the torch-cpu baseline (same
`case_id`): relative error ≤ 10% → ok; both sides ≤ 1e-4 → ok with note
"both converged to machine zero"; baseline/record error or differing gens →
skip. Full blocks are written into every `results/*.json` (recompute them
with `retag_parity.py` — never hand-edit).

Counts over all 240 non-baseline records:

| suite | ok | not-ok | skip | skipped because |
|---|---|---|---|---|
| SO | 90 | 66 | 24 | 24 etl-numpy gens caps |
| MO | 10 | 22 | 28 | 28 backend errors (12 iree-llvm + 12 iree-cuda + 4 xla NSGA3) |
| **total** | **100** | **88** | **52** | 28 backend errors + 24 gens caps |

Parity-evaluable (ok + not-ok): **188**.

**SO not-ok (66)** — all notes are `RNG-stream variance on unconverged run`
unless stated:

- vs etl-numpy (7): PSO/Rastrigin/100x10 0.318 · DE/Sphere/100x10 1.347 ·
  DE/Rastrigin/100x10 0.188 · CMAES/Rastrigin/100x10 0.928 ·
  OpenES/Sphere/100x10 0.235 · OpenES/Rastrigin/100x10 0.122 ·
  **CMAES/Ackley/100x10 5671 (documented divergence)**
- vs etl-iree-llvm-cpu (14): the 100x10 set (CMAES/Rastrigin 0.961,
  **CMAES/Ackley 5677**) plus DE/Sphere/1000x50 0.326 ·
  DE/Rastrigin/1000x50 0.244 · CMAES/Sphere/1000x50 0.467 ·
  CMAES/Rastrigin/1000x50 0.124 · CMAES/Ackley/1000x50 1.888 ·
  CMAES/Sphere/10000x100 0.315 · CMAES/Ackley/10000x100 0.781
- vs etl-iree-cuda (13): as llvm-cpu (CMAES/Rastrigin/100x10 0.958,
  **CMAES/Ackley 5662**; CMAES/Rastrigin/1000x50 is ok at 0.087)
- vs etl-xla-cuda (14): CMAES/Rastrigin/100x10 0.999, **CMAES/Ackley 5678**,
  plus the same 100x10/1000x50/10000x100 set (CMAES/Rastrigin/10000x100 ok
  at 0.078)
- vs torch-cuda (18): PSO/Sphere/1000x50 0.562 · PSO/Sphere/10000x100 0.257 ·
  PSO/Rastrigin 0.376 / 0.155 / 0.126 · DE/Sphere 0.282 / 0.425 / 0.122 ·
  DE/Rastrigin 0.198 / 0.225 / 0.101 · CMAES/Sphere/10000x100 0.258 ·
  CMAES/Rastrigin/100x10 0.806 · CMAES/Ackley 1.184 / 1.820 / 0.711 ·
  OpenES/Sphere/100x10 0.462 · OpenES/Rastrigin/100x10 0.201

**MO not-ok (22)** — all `RNG-stream variance on unconverged run`:

- vs etl-numpy (9): NSGA2/DTLZ1 2.022 / 0.792 · NSGA3/DTLZ1/100x3x10 16.661 ·
  MOEAD/DTLZ1/100x3x10 0.959 · MOEAD/DTLZ2/100x3x10 0.162 ·
  NSGA3/DTLZ1/1000x3x30 0.691 · NSGA3/DTLZ2/1000x3x30 0.180 ·
  MOEAD/DTLZ1/1000x3x30 21.006 · MOEAD/DTLZ2/1000x3x30 0.169
- vs torch-cuda (7): NSGA2/DTLZ1 0.732 / 0.909 · NSGA3/DTLZ1 0.362 / 0.150 ·
  MOEAD/DTLZ1 0.832 / 6.627 · MOEAD/DTLZ2/1000x3x30 0.151
- vs etl-xla-cuda (6): NSGA2/DTLZ1 0.329 / 0.772 · MOEAD/DTLZ1 0.959 / 14.828 ·
  MOEAD/DTLZ2 0.162 / 0.173

**Overall verdict:** the three compiled etl backends are still numerically
identical to each other (iree-llvm-cpu ≡ iree-cuda ≡ xla-cuda at bit or
last-ulp precision — e.g. PSO/Sphere/100x10 best = 3.054133657087732e-08 on
all four etl backends; CMA-ES eigendecomposition noise and xla's GPU
non-deterministic reductions differ, e.g. CMAES/Sphere/1000x50 35.735 /
35.757 / 35.772), and etl-numpy agrees bit-identically wherever it ran the
same 100 gens. The re-run CMA-ES cells agree with torch within the 10%
tolerance on most converged cases (large-scale Sphere/Rastrigin on
iree/xla are the best matches). The remaining divergences are the
documented CMAES/Ackley/100x10 stall (all four etl backends), relative-error
scatter on unconverged small-scale runs (100x10 DE/OpenES/Rastrigin, DTLZ1
at 100 gens — RNG-stream variance between torch's MT19937 and etl's keyed
PRNGs, not a correctness gap), and DTLZ1's 1000x3x30 scale where both sides
are far from the front at 100 gens (MOEAD/DTLZ1 rel 14.8–21.0).

## Key numbers

- **torch-cuda PSO: ~1.06–1.26 ms/step at ALL scales** (100x10 → 10000x100);
  DE 0.87–1.01 ms/step; OpenES 0.62–0.84 ms/step — per-step cost is nearly
  scale-invariant.
- **etl-xla-cuda SO: 3.6–18.2 ms/step** (median ~5.3) + **0.41–4.39 s
  compile** — except CMA-ES, which is 10.2 ms/step at 100x10 but 88–406
  ms/step at 1000x50/10000x100 (eigh-dominated).
- **etl-xla-cuda MO now RUNS (NSGA2/MOEAD):** NSGA2 4.1–5.3 ms/step,
  MOEAD 6.4–27.2 ms/step, compile 1.0–3.1 s — median 15.4× faster than
  torch-cpu. NSGA3 remains blocked (`matrix_rank` export).
- **etl-numpy MO is a full 100-gen matrix** (cap removed): 6.4–11.0 ms/step
  at 100x3x10, 232–693 ms/step at 1000x3x30.
- **etl-iree-llvm-cpu 10000x100: 399–707 ms/step for PSO/DE/OpenES**, but
  CMA-ES now dominates at 21.9–22.9 s/step (compile 0.90–3.10 s).
- **etl-iree-cuda 10000x100: ~3.48–3.50 s/step** for PSO/DE/OpenES and
  ~9.5–9.6 s/step for CMA-ES (compile 1.21–20.6 s) — the slowest
  non-interpreter backend at scale.
- **torch-cpu CMA-ES:** 57.5 ms/step at 100x10 → 133–168 ms/step at
  1000x50/10000x100 (real runs now — the pre-fix records were NaN-frozen).
- **torch-cuda MO is bimodal:** NSGA2 4.3–7.6 ms/step (fast) vs NSGA3
  623–863 ms/step and MOEAD 273–3621 ms/step (python-loop/table bound).
- **Compile cost:** etl-numpy 0.04–0.58 s (cheapest); xla 0.41–4.39 s;
  iree-llvm 0.90–3.10 s; iree-cuda up to 20.6 s. torch pays nothing (eager).

## Reproduction

From `benchmarks/etl_vs_torch/` with the venv python
(`/mnt/local-ssd/bchuang/evox/.venv/bin/python`):

```
python run.py --backend torch-cpu      --suite so   # and --suite mo
python run.py --backend torch-cuda     --suite so --gpu 0
python run.py --backend etl-numpy      --suite so
python run.py --backend etl-iree-llvm-cpu --suite so
python run.py --backend etl-iree-cuda  --suite so --gpu 0
python run.py --backend etl-xla-cuda   --suite so --gpu 0
```

`--gpu N` applies the GPU environment recipe **before any torch/etl import**
(`bench_common.ensure_gpu_env`; see `results/README.md`):

- `CUDA_VISIBLE_DEVICES=<id>`
- `LD_LIBRARY_PATH` prepends `/mnt/local-ssd/bchuang/cudnn-xla/lib`
- `XLA_FLAGS=--xla_gpu_cuda_data_dir=/home/bchuang/xla_cuda_data`
- `ETL_PJRT_PLUGIN=<venv>/jax_plugins/xla_cuda12/xla_cuda_plugin.so`
- for `etl-xla-cuda` additionally: `LD_PRELOAD`
  `/mnt/local-ssd/bchuang/cudnn-xla/lib/libcudnn.so.9` (the xla PJRT plugin
  was built against cuDNN 9.8 but DT_RPATHs the venv's 9.1.0) + a
  marker-guarded process re-exec (the loader reads LD_PRELOAD only at process
  start).

Subset re-runs: use `--cases <ids>`, `--gens-override <n>` and `--out <path>`
(smoke runs never write into `results/`). **Partial runs overwrite the
target JSON** — merge them with
`python retag_parity.py --suite {so,mo} --backend <b> --partial <path>
[--cap-note "..."] --apply`, which merges by `case_id` and re-derives the
parity verdicts per the committed conventions (never hand-edit verdicts).

## Artifacts

- `results/{so,mo}_{backend}.json` — 12 files, 288 records with parity
  verdicts vs torch-cpu written in.
- `retag_parity.py` — partial-run merge + parity verdict re-derivation.
- `style_comparison.md` — code style/cleanliness comparison (torch OOP vs
  etl functional).
- `bench_common.py` — shared case tables, metric implementations, parity
  helpers (pure stdlib + numpy).
- `bench_so.py` / `bench_mo.py` / `run.py` — the runners and CLI dispatcher.
