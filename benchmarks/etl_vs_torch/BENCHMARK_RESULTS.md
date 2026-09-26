# evox (torch) vs evox_etl — Benchmark Results

Comparison of the torch OOP `evox` (`src/evox/`) against the functional
`evox_etl` (`src/evox_etl/`). Six backends, two suites (single-objective,
multi-objective), small → large scale. All numbers below are read from the
committed `results/*.json` (12 files, one per suite×backend).

## Measurement caveats (2026-09-07)

- The committed etl-xla-cuda SO ms/step figures were produced by an old
  harness mapping bug: `bench_common.etl_backend_spec` returned
  `("xla", None)`, which builds a CPU-kind executable that host-stages every
  input on every step.
- The committed xla figures (5.73 / 17.0 ms/step at 1000x50 / 10000x100)
  were reproduced exactly with host staging on a healthy GPU, while the same
  code device-resident runs at ~0.9-1.0 ms/step, so they are obsolete as
  GPU-performance evidence.
- The old committed etl-iree-cuda SO numbers (35.70 / 3482 ms/step at
  1000x50 / 10000x100) were broken-GPU-era artifacts; both iree SO files
  (`so_etl-iree-cuda.json`, `so_etl-iree-llvm-cpu.json`) have since been
  fully refreshed (36/36 records each) under iree 3.11.0 on healthy
  hardware, and git history preserves the old artifacts. The refreshed
  CMAES cells are slow for the known lowering reason in *CMAES slow path*
  below — not an environment artifact.
- The harness mapping is fixed in `bench_common.py`
  (`etl_backend_spec("etl-xla-cuda")` now returns `("xla", "cuda:0")`), and
  `StdWorkflow._finish_init` now activates the workflow's backend adapter
  before placing state on device (commit 1a6fef1c), so device-resident xla
  runs no longer need manual adapter pre-activation.
- Fresh measurements below (2026-09-07, idle RTX A6000 #6, seed 42, 2 warmup
  steps + 100 timed gens, compile time excluded; scratch records written to
  `$TMPDIR` only — no committed `results/*.json` was modified; convergence
  reproduces the committed best-fitness values exactly on both backends):

| case | torch-cuda ms/step | etl-xla-cuda ms/step | xla/torch ratio |
|---|---|---|---|
| PSO/Sphere/1000x50 | 1.150 | 0.896 | 0.78× |
| PSO/Sphere/10000x100 | 1.217 | 0.988 / 1.045 (2 samples) | 0.81-0.86× |
| DE/Rastrigin/1000x50 | 1.036 | 0.799 | 0.77× |
| DE/Rastrigin/10000x100 | 1.127 | 0.776 / 0.843 (2 samples) | 0.69-0.75× |

- etl-xla-cuda is therefore 1.16-1.45× faster than torch-cuda per step on
  these cells, at identical convergence to the committed records.

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
| torch-cpu | 1.538e-08 / 0.677 | 8.088 / 0.6997 | 20.01 / 0.7587 | 0.2588 / 0.5298 | 47.47 / 0.5681 | 20 / 0.6173 | 2.064e-07 / 59.09 | 26.12 / 123.1 | 0.003625 / 132.4 | 172.8 / 0.4103 | 252.1 / 0.4486 | 20.66 / 0.4905 |
| torch-cuda | 9.313e-09 / 1.063 | 5.048 / 1.156 | 20 / 1.198 | 0.186 / 0.8846 | 56.9 / 0.9457 | 19.42 / 1.004 | 6.428e-08 / 58.55 | 5.055 / 119 | 0.007918 / 124.2 | 92.91 / 0.6223 | 201.4 / 0.6924 | 20.5 / 0.7366 |
| etl-numpy | 3.054e-08 / 0.8298 | 5.52 / 0.9024 | 20 / 0.9534 | 0.6075 / 0.9725 | 56.38 / 1.038 | 18.95 / 1.083 | 9.772e-09 / 1.312 | 21.31 / 1.346 | 0.0006533 / 1.496 | 132.1 / 0.4907 | 221.2 / 0.5614 | 20.69 / 0.6164 |
| etl-iree-llvm-cpu | 3.054e-08 / 0.5004 | 5.52 / 0.6652 | 20 / 0.8077 | 0.6075 / 0.4589 | 56.38 / 0.7068 | 18.95 / 0.5174 | 9.269e-09 / 21.98 | 7.57 / 24.76 | 0.003652 / 24.62 | 132.1 / 0.6767 | 221.2 / 0.6438 | 20.69 / 0.7569 |
| etl-iree-cuda | 3.054e-08 / 0.7115 | 5.52 / 0.916 | 20 / 0.9245 | 0.6075 / 0.964 | 56.38 / 0.5291 | 18.95 / 0.5545 | 9.582e-09 / 41.91 | 27.85 / 46.5 | 0.001409 / 44.96 | 132.1 / 0.7364 | 221.2 / 0.7646 | 20.69 / 0.8201 |
| etl-xla-cuda | 3.054e-08 / 5.366 | 5.52 / 5.112 | 20 / 5.081 | 0.6075 / 3.271 | 56.38 / 3.914 | 19.83 / 4.215 | 1.05e-08 / 9.937 | 27.28 / 9.167 | 0.0005455 / 9.522 | 132.1 / 4.78 | 221.2 / 4.817 | 20.69 / 4.705 |

### 1000 × 50

| backend | PS | PR | PA | DS | DR | DA | CS | CR | CA | OS | OR | OA |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-cpu | 30.77 / 2.558 | 391.5 / 2.646 | 20 / 2.693 | 1.798e+04 / 2.223 | 2.244e+04 / 2.608 | 20.28 / 2.419 | 67.05 / 130.1 | 489.7 / 136.2 | 7.317 / 141.7 | 1814 / 0.8695 | 2271 / 0.9689 | 21.26 / 1.005 |
| torch-cuda | 13.46 / 1.085 | 330.6 / 1.143 | 20.05 / 1.205 | 2.562e+04 / 0.9 | 2.749e+04 / 0.9707 | 20.26 / 1.028 | 68.77 / 120.3 | 530.2 / 124.2 | 20.63 / 125.9 | 1751 / 0.6802 | 2214 / 0.7842 | 21.21 / 0.8361 |
| etl-numpy | 201.4@50g / 2.166 | 875.2@50g / 2.717 | 20.58@50g / 2.739 | 4.271e+04@50g / 1.913 | 4.2e+04@50g / 1.904 | 20.52@50g / 1.983 | 1.109e+04@50g / 3.402 | 1.163e+04@50g / 3.467 | 20.3@50g / 3.821 | 1.306e+04@50g / 0.9856 | 1.365e+04@50g / 1.111 | 21.24@50g / 1.202 |
| etl-iree-llvm-cpu | 31.94 / 1.281 | 365.9 / 1.342 | 20.56 / 1.283 | 2.384e+04 / 1.059 | 2.792e+04 / 1.01 | 20.14 / 1.361 | 88.41 / 1257 | 522.8 / 1223 | 21.13 / 1324 | 1887 / 1.259 | 2370 / 1.484 | 21.24 / 1.541 |
| etl-iree-cuda | 31.94 / 0.8862 | 365.9 / 0.8956 | 20.56 / 1.116 | 2.384e+04 / 0.811 | 2.792e+04 / 0.7118 | 20.14 / 0.5871 | 88.18 / 1481 | 527.9 / 1489 | 10.85 / 1537 | 1887 / 0.6327 | 2370 / 0.761 | 21.24 / 0.9365 |
| etl-xla-cuda | 31.94 / 5.73 | 365.9 / 5.738 | 20.56 / 5.291 | 2.384e+04 / 4.36 | 2.792e+04 / 4.602 | 20.08 / 3.638 | 88.07 / 88.07 | 540.1 / 88.48 | 18.57 / 87.44 | 1887 / 3.71 | 2370 / 3.836 | 21.24 / 3.772 |

### 10000 × 100

| backend | PS | PR | PA | DS | DR | DA | CS | CR | CA | OS | OR | OA |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-cpu | 2548 / 27.45 | 3846 / 36.21 | 20.08 / 26.65 | 9.718e+04 / 16.77 | 1.08e+05 / 20.08 | 20.61 / 17.7 | 294.1 / 158 | 1261 / 163.8 | 11.91 / 165 | 4219 / 8.587 | 5163 / 8.237 | 21.3 / 10.43 |
| torch-cuda | 1893 / 1.085 | 3362 / 1.135 | 20.12 / 1.259 | 1.091e+05 / 0.8736 | 1.189e+05 / 1.01 | 20.67 / 1.002 | 218.2 / 122.8 | 1189 / 127.2 | 20.38 / 128.9 | 4068 / 0.6265 | 5095 / 0.7006 | 21.31 / 0.754 |
| etl-numpy | 8.43e+04@10g / 73.99 | 8.52e+04@10g / 66.56 | 20.35@10g / 64.45 | 1.595e+05@10g / 42.13 | 1.555e+05@10g / 38.57 | 20.82@10g / 42.41 | 2.123e+05@10g / 53.57 | 2.133e+05@10g / 57.43 | 21.34@10g / 48.73 | 1.484e+05@10g / 26.79 | 1.492e+05@10g / 25.36 | 21.39@10g / 22.31 |
| etl-iree-llvm-cpu | 2691 / 15.39 | 4119 / 15.6 | 20.02 / 19.08 | 1.048e+05 / 15.49 | 1.083e+05 / 18.79 | 20.56 / 27.54 | 350.6 / 2.225e+04 | 1325 / 2.227e+04 | 17.17 / 2.23e+04 | 4112 / 18.74 | 5110 / 21.29 | 21.32 / 20.38 |
| etl-iree-cuda | 2691 / 3.486 | 4119 / 3.531 | 20.02 / 3.531 | 1.048e+05 / 2.131 | 1.083e+05 / 1.778 | 20.56 / 1.762 | 348.5 / 6902 | 1326 / 7275 | 20.59 / 7400 | 4112 / 2.089 | 5110 / 2.23 | 21.32 / 2.139 |
| etl-xla-cuda | 2691 / 17 | 4119 / 15.2 | 20.02 / 18.16 | 1.048e+05 / 11.18 | 1.083e+05 / 11.6 | 20.64 / 11.6 | 348.5 / 404.6 | 1323 / 399.3 | 14.62 / 397.2 | 4112 / 8.234 | 5110 / 9.325 | 21.32 / 9.34 |

### SO compile time and speedup

Compile time range over successful cases (`wf.init(seed=42)`; all 36 per
etl backend — no SO compile failures remain):

| backend | SO compile time |
|---|---|
| torch-cpu / torch-cuda | 0.0 s (eager) |
| etl-numpy | 0.044 – 0.382 s |
| etl-iree-llvm-cpu | 0.881 – 2.945 s |
| etl-iree-cuda | 0.851 – 2.797 s |
| etl-xla-cuda | 0.411 – 4.392 s |

ms/step ratio, case-matched, median (min – max); **>1 = faster than the baseline**:

| backend | vs torch-cpu | vs torch-cuda |
|---|---|---|
| torch-cuda | 1.28× (0.60× – 31.9×) | — |
| etl-numpy | 0.84× (0.32× – 91.4×) | — |
| etl-iree-llvm-cpu | 0.996× (0.0071× – 5.38×) | — |
| etl-iree-cuda | 1.39× (0.022× – 11.3×) | 0.90× (0.017× – 2.76×) |
| etl-xla-cuda | 0.51× (0.086× – 13.9×) | 0.21× (0.064× – 13×) |

The large "etl wins" outliers are CMA-ES 100x10 — torch's 100x10 CMA-ES CPU
path is slow (~59–132 ms/step vs 1.3–1.5 on etl-numpy, 9.2–9.9 on xla-cuda,
22–25 on iree-llvm-cpu, 42–47 on iree-cuda) — plus torch-cuda's 31.9×
PSO/Rastrigin/10000x100 and iree-cuda's 11.3× DE/Rastrigin/10000x100. The
"torch wins" floor is set by large-scale CMA-ES on iree: its per-step
covariance `etl.eigh` lowers to an O(n^4) cyclic-Jacobi while-loop (see
*CMAES slow path* below), so iree-llvm-cpu runs CMA-ES 10000x100 at
22.2–22.3 s/step and iree-cuda at 6.9–7.4 s/step vs torch-cpu's 0.16 s/step.
That floor drags the all-cell medians down (iree-cuda 1.39×, iree-llvm-cpu
0.996× vs torch-cpu); excluding CMA-ES the medians rise to 2.74× (iree-cuda)
and 1.07× (iree-llvm-cpu), max 11.3×. At 10000x100 iree-cuda beats torch-cpu
on every non-CMA-ES cell (1.76–3.53 vs 8.2–36.2 ms/step, 3.7–11.3×);
iree-llvm-cpu wins on PSO (1.4–2.3×) and DE/Sphere+Rastrigin (≈1.1×) but
loses on DE/Ackley (0.64×) and OpenES (0.39–0.51×). torch-cuda remains the
fastest GPU path at that scale (0.63–1.26 ms/step), with xla-cuda next at
8.2–18.2 ms/step.

## CMAES slow path (iree)

All nine CMAES cells in each of `so_etl-iree-llvm-cpu.json` and
`so_etl-iree-cuda.json` carry a record-level note: per-step covariance
`etl.eigh` (pop_dim × pop_dim) lowers to an iree cyclic-Jacobi while-loop
composition — stablehlo 1.0 / iree 3.11.0 ship no native `stablehlo.eigh`
(only cholesky survives) — and that composition is O(n^4) in pop_dim. The
cells ran healthy (no error records); the note documents the slowness and
does not change their parity verdicts.

Measured on healthy hardware with iree 3.11.0:

- Standalone `etl.eigh` on iree-cuda: 39.0 / 1317 / 5778 ms at dims
  10 / 50 / 100, while the no-eigh control runs at 0.087 ms/step. On
  iree-llvm-cpu the dim-100 eigh takes 23676 ms (control 0.157 ms/step).
- Real-harness iree-cuda (gens=10) matches the standalone cost: 39.5 /
  1036.6 / 5779.3 ms/step at the same dims.
- xla-cuda runs the same cases at 387.3 ms/step (native eigh) — 15–18×
  faster.
- Verdict: an iree lowering property — not an evox_etl graph defect, not a
  host round-trip, and not a measurement artifact.
- Upstream documentation: `/mnt/local-ssd/bchuang/etl/etl/backends/stablehlo/CONTEXT.md`.

## Multi-objective (MO) — NSGA2 / NSGA3 / MOEAD × DTLZ1 / DTLZ2

Cell = `hv/igd / ms_per_step`. D1/D2 = DTLZ1/DTLZ2. `err` = error record
(see *Caps & limits*). MOEAD's effective population is the Das-Dennis vector
count (pop_size is overwritten on both sides).

### 100 × 3 × 10

| backend | NSGA2/1 | NSGA2/2 | NSGA3/1 | NSGA3/2 | MOEAD/1 | MOEAD/2 |
|---|---|---|---|---|---|---|
| torch-cpu | 7.92/2.045 / 4.2 | 7.46/0.02443 / 2.995 | 6.422/1.804 / 407.5 | 7.461/0.02572 / 420.3 | 184.1/4.505 / 97.02 | 7.46/0.02888 / 99.17 |
| torch-cuda | 7.685/0.5479 / 5.126 | 7.458/0.02459 / 4.291 | 6.456/1.151 / 623 | 7.463/0.02535 / 625.5 | 30.97/2.727 / 273.1 | 7.456/0.03117 / 279.5 |
| etl-numpy | 23.94/3.999 / 7.668 | 7.462/0.02312 / 6.413 | 113.4/9.624 / 11.21 | 7.461/0.02572 / 8.514 | 7.562/0.5722 / 61.36 | 7.455/0.03355 / 60.34 |
| etl-iree-llvm-cpu | err | err | err | err | err | err |
| etl-iree-cuda | err | err | err | err | err | err |
| etl-xla-cuda | 10.52/1.847 / 4.784 | 7.459/0.02465 / 4.146 | 113.3/9.927 / 4.771 | 7.462/0.02549 / 3.899 | 7.562/0.5722 / 6.432 | 7.455/0.03355 / 6.458 |

### 1000 × 3 × 30

| backend | NSGA2/1 | NSGA2/2 | NSGA3/1 | NSGA3/2 | MOEAD/1 | MOEAD/2 |
|---|---|---|---|---|---|---|
| torch-cpu | 3.959e+04/51.1 / 83.07 | 7.463/0.01669 / 45.22 | 1.706e+04/41.36 / 417.9 | 7.466/0.01521 / 387.1 | 465/6.829 / 1033 | 7.459/0.0208 / 1028 |
| torch-cuda | 3616/23.1 / 7.646 | 7.464/0.01624 / 5.489 | 1.962e+04/44.2 / 863.3 | 7.467/0.01478 / 719.1 | 3546/15.03 / 3505 | 7.462/0.01766 / 3621 |
| etl-numpy | 8241/32.67 / 289.7 | 7.462/0.01756 / 231.7 | 5269/30.06 / 321.3 | 7.463/0.01795 / 260.2 | 1.023e+04/4.718 / 692.9 | 7.454/0.02432 / 688.6 |
| etl-iree-llvm-cpu | err | err | err | err | err | err |
| etl-iree-cuda | err | err | err | err | err | err |
| etl-xla-cuda | 9039/34.23 / 4.654 | 7.462/0.01713 / 5.268 | 8048/33.65 / 5.355 | 7.465/0.01629 / 5.010 | 7360/4.716 / 27.15 | 7.454/0.0244 / 26.92 |

### MO compile time and speedup

| backend | MO compile time |
|---|---|
| torch-cpu / torch-cuda | 0.0 s (eager) |
| etl-numpy | 0.346 – 0.582 s |
| etl-xla-cuda | 1.045 – 3.057 s (all 12 cases) |
| etl-iree-llvm-cpu / etl-iree-cuda | — (no case compiles; see *Caps & limits*) |

| backend | vs torch-cpu |
|---|---|
| torch-cuda | 0.65× (0.28× – 10.9×) |
| etl-numpy | 1.49× (0.20× – 51.0×) |
| etl-xla-cuda | 28.0× (0.72× – 107.8×; all 12 cases) |
| etl-iree-llvm-cpu / etl-iree-cuda | — (all err) |

etl-numpy's 51× outlier is NSGA3/DTLZ2 100x3x10 (8.2 vs 420 ms/step) —
torch NSGA3's CPU path is python-loop-bound. etl-xla-cuda's 28.0× median
reflects compiled NSGA2/NSGA3/MOEAD (3.9–27.2 ms/step) vs torch's
python-bound NSGA3/MOEAD CPU paths (99–1033 ms/step); the largest speedups
are NSGA3's 77–108× (torch-cpu NSGA3 runs at 387–420 ms/step, xla-cuda at
3.9–5.4 ms/step). torch-cuda stays bimodal (NSGA2 5.5–7.6 ms/step fast;
NSGA3 623–863, MOEAD 273–3621 ms/step slow).

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
  compiles on xla-cuda (NSGA2+MOEAD run end-to-end). Two blockers
  remain, recorded with exact errors in the JSONs:
  1. **NSGA3 `matrix_rank`/`solve` export rejection — FIXED.**
     `src/evox_etl/algorithms/mo/nsga3.py` now uses eigh-based equivalents
     (full-rank guard via sqrt(eigvalsh(AᵀA)) accumulated in float64;
     hyperplane solve via normal equations). NSGA3 now runs end-to-end on
     etl-numpy (hv/igd bit-identical to the pre-fix records) and
     etl-xla-cuda (all 4 cells: 3.9–5.4 ms/step; DTLZ2 parity ok at rel
     0.009/0.071).
  2. **iree-compile segfault on all MO programs (23/24 iree cells):**
     `BackendError: iree-compile failed to compile ... Error code: -11` with
     a stack dump inside `libIREECompiler.so` (iree-cuda 12 cells;
     iree-llvm-cpu 11 cells). The export succeeds; the compiler itself
     crashes (deep recursion in the stack trace). NSGA3 was previously
     stopped at export; post-fix it passes export and crashes in the same
     compiler bug — an upstream iree while-loop compiler issue (the
     unmodified NSGA2/MOEAD tell triggers it identically). Not retryable.
  3. **MOEAD/DTLZ2/100x3x10 on iree-llvm-cpu** is the only iree MO cell that
     passes export AND compile but dies at runtime: `ValueError: ... ref is
     null ... hal.buffer_view.create` (INVALID_ARGUMENT).
- **CMAES/Ackley/100x10 divergence — FIXED.** The stall was a port
  divergence in `src/evox_etl`: the port used the canonical `p_c` rank-one
  outer product while torch's 1-D `p_c @ p_c.T` is a scalar dot product
  (isotropic c_1 inflation into every element of C). With the torch quirk
  replicated, all four etl backends now escape Ackley's zero-gradient
  plateau like torch: best 0.00055 (xla-cuda) / 0.00065 (numpy) / 0.0014
  (iree-cuda) / 0.0037 (iree-llvm-cpu) vs torch-cpu 0.0036 (iree-llvm-cpu
  parity ok at rel 0.007; the others rel 0.61–0.85, RNG-stream scatter).
  Large-scale Ackley cells (1000x50/10000x100) remain unconverged on
  torch-cuda (≈20.6/20.4) and scatter on the etl compiled backends (rel
  0.44–1.89) — RNG-stream variance on unconverged Ackley runs, not the
  quirk stall.
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
| SO | 94 | 62 | 24 | 24 etl-numpy gens caps |
| MO | 12 | 24 | 24 | 24 backend errors (12 iree-llvm-cpu + 12 iree-cuda) |
| **total** | **106** | **86** | **48** | 24 backend errors + 24 gens caps |

Parity-evaluable (ok + not-ok): **192**.

**SO not-ok (62)** — all notes are `RNG-stream variance on unconverged run`
unless stated:

- vs etl-numpy (7): PSO/Rastrigin/100x10 0.318 · DE/Sphere/100x10 1.347 ·
  DE/Rastrigin/100x10 0.188 · CMAES/Rastrigin/100x10 0.184 ·
  CMAES/Ackley/100x10 0.820 · OpenES/Sphere/100x10 0.236 ·
  OpenES/Rastrigin/100x10 0.122
- vs etl-iree-llvm-cpu (12): the 100x10 set (CMAES/Rastrigin 0.710) plus
  DE/Sphere/1000x50 0.326 · DE/Rastrigin/1000x50 0.244 ·
  CMAES/Sphere/1000x50 0.319 · CMAES/Ackley/1000x50 1.887 ·
  CMAES/Sphere/10000x100 0.192 · CMAES/Ackley/10000x100 0.442
- vs etl-iree-cuda (12): the 100x10 set (CMAES/Ackley 0.611) plus
  DE/Sphere/1000x50 0.326 · DE/Rastrigin/1000x50 0.244 ·
  CMAES/Sphere/1000x50 0.315 · CMAES/Ackley/1000x50 0.483 ·
  CMAES/Sphere/10000x100 0.185 · CMAES/Ackley/10000x100 0.728
- vs etl-xla-cuda (13): CMAES/Rastrigin/1000x50 0.103 · CMAES/Ackley
  0.850 / 1.538 / 0.228 · plus the same 100x10/1000x50/10000x100 set
  (CMAES/Sphere/1000x50 0.314, CMAES/Sphere/10000x100 0.185)
- vs torch-cuda (18): PSO/Sphere/1000x50 0.562 · PSO/Sphere/10000x100 0.257 ·
  PSO/Rastrigin 0.376 / 0.155 / 0.126 · DE/Sphere 0.282 / 0.425 / 0.122 ·
  DE/Rastrigin 0.198 / 0.225 / 0.101 · CMAES/Sphere/10000x100 0.258 ·
  CMAES/Rastrigin/100x10 0.806 · CMAES/Ackley 1.184 / 1.820 / 0.711 ·
  OpenES/Sphere/100x10 0.462 · OpenES/Rastrigin/100x10 0.201

**MO not-ok (24)** — all `RNG-stream variance on unconverged run`:

- vs etl-numpy (9): NSGA2/DTLZ1 2.022 / 0.792 · NSGA3/DTLZ1/100x3x10 16.661 ·
  MOEAD/DTLZ1/100x3x10 0.959 · MOEAD/DTLZ2/100x3x10 0.162 ·
  NSGA3/DTLZ1/1000x3x30 0.691 · NSGA3/DTLZ2/1000x3x30 0.180 ·
  MOEAD/DTLZ1/1000x3x30 21.006 · MOEAD/DTLZ2/1000x3x30 0.169
- vs torch-cuda (7): NSGA2/DTLZ1 0.732 / 0.909 · NSGA3/DTLZ1 0.362 / 0.150 ·
  MOEAD/DTLZ1 0.832 / 6.627 · MOEAD/DTLZ2/1000x3x30 0.151
- vs etl-xla-cuda (8): NSGA2/DTLZ1 0.329 / 0.772 · NSGA3/DTLZ1 16.636 / 0.528 ·
  MOEAD/DTLZ1 0.959 / 14.828 · MOEAD/DTLZ2 0.162 / 0.173

**Overall verdict:** the three compiled etl backends are still numerically
identical to each other (iree-llvm-cpu ≡ iree-cuda ≡ xla-cuda at bit or
last-ulp precision — e.g. PSO/Sphere/100x10 best = 3.054133657087732e-08 on
all four etl backends; CMA-ES eigendecomposition noise and xla's GPU
non-deterministic reductions differ, e.g. CMAES/Sphere/1000x50 88.412 /
88.175 / 88.074), and etl-numpy agrees bit-identically wherever it ran the
same 100 gens. The re-run CMA-ES cells agree with torch within the 10%
tolerance on most converged cases (large-scale Sphere/Rastrigin on
iree/xla are the best matches), and CMAES/Ackley/100x10 now converges to
the torch region on all four etl backends (best 0.00055–0.0037 vs torch-cpu
0.0036). The remaining divergences are RNG-stream scatter on unconverged
runs — large-scale Ackley 1000x50/10000x100 (torch-cuda and the etl
compiled backends), small-scale 100x10 DE/OpenES/Rastrigin, and DTLZ1 MO at
100 gens (torch's MT19937 vs etl's keyed PRNGs, not a correctness gap) —
plus DTLZ1's 1000x3x30 scale where both sides are far from the front at 100
gens (MOEAD/DTLZ1 rel 14.8–21.0). On the MO side, NSGA3 is unblocked
post-fix: etl-xla-cuda now runs the full 12-cell MO matrix (NSGA3 3.9–5.4
ms/step; DTLZ2 parity ok at rel 0.009/0.071; DTLZ1 remains RNG-stream
scatter on unconverged runs, matching etl-numpy). iree stays fully blocked
by the upstream iree-compile segfault (23/24 MO cells; the remaining
iree-llvm-cpu cell dies at runtime with the `ref is null` INVALID_ARGUMENT).

## Key numbers

- **torch-cuda PSO: ~1.06–1.26 ms/step at ALL scales** (100x10 → 10000x100);
  DE 0.87–1.01 ms/step; OpenES 0.62–0.84 ms/step — per-step cost is nearly
  scale-invariant.
- **etl-xla-cuda SO: 3.6–18.2 ms/step** (median ~5.3) + **0.41–4.39 s
  compile** — except CMA-ES, which is 9.2–9.9 ms/step at 100x10 but 87–405
  ms/step at 1000x50/10000x100 (eigh-dominated).
- **etl-xla-cuda MO is a full 12-case matrix:** NSGA2 4.1–5.3 ms/step,
  NSGA3 3.9–5.4 ms/step, MOEAD 6.4–27.2 ms/step, compile 1.0–3.1 s —
  median 28.0× faster than torch-cpu (NSGA3 alone 77–108×: torch's CPU
  NSGA3 path is python-loop-bound).
- **etl-numpy MO is a full 100-gen matrix** (cap removed): 6.4–11.2 ms/step
  at 100x3x10, 232–693 ms/step at 1000x3x30.
- **etl-iree-llvm-cpu 10000x100: 15.4–27.5 ms/step for PSO/DE/OpenES**, but
  CMA-ES dominates at 22.25–22.30 s/step (compile 0.88–2.95 s).
- **etl-iree-cuda 10000x100: 1.76–3.53 ms/step for PSO/DE/OpenES** (3.7–11.3×
  faster than torch-cpu) and 6.90–7.40 s/step for CMA-ES — the O(n^4) iree
  eigh lowering (see *CMAES slow path*).
- **torch-cpu CMA-ES:** 59–132 ms/step at 100x10 → 130–165 ms/step at
  1000x50/10000x100 (real runs now — the pre-fix records were NaN-frozen).
- **torch-cuda MO is bimodal:** NSGA2 4.3–7.6 ms/step (fast) vs NSGA3
  623–863 ms/step and MOEAD 273–3621 ms/step (python-loop/table bound).
- **Compile cost:** etl-numpy 0.04–0.58 s (cheapest); xla 0.41–4.39 s;
  iree-llvm-cpu 0.88–2.95 s; iree-cuda 0.85–2.80 s. torch pays nothing
  (eager).

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
