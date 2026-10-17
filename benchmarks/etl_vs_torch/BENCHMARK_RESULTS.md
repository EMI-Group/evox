# evox (torch) vs evox_etl — Benchmark Results

Comparison of the torch OOP `evox` (`src/evox/`) against the functional
`evox_etl` (`src/evox_etl/`). Two suites (single-objective, multi-objective),
small → large scale, on two hosts: six backends on an NVIDIA A6000 host and
the ROCm backends on an AMD Instinct MI300X (see *Post-fix ROCm re-run*). All
numbers below are read from the committed `results/*.json` — 12
per-suite×backend files for the A6000 matrix, plus the ROCm-round files listed
under *Artifacts*.

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
- **The device-resident etl rows of the SO/MO tables below are conservative.**
  They predate the five `evox_etl` changes listed under *Post-fix ROCm re-run* → *What changed* (batched `eigh`, split rank-0 reduction, faster XLA per-call hot path, emitter correctness fixes, pipelined monitor readback).
  Where those fixes bite, their ms/step is ~1.2-1.6× slow — *Post-fix ROCm re-run* carries the measured old→new deltas.
  The `etl-numpy` column is unaffected by the readback win but does not reflect the etl fixes either.

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

## Post-fix ROCm re-run (AMD Instinct MI300X, ROCm 7.1)

This is the **first committed ROCm matrix** (`torch-rocm` /
`etl-xla-rocm` / `etl-iree-rocm` / `etl-iree-llvm-cpu_mi300x` result files
below), run to measure the etl fixes against the previous round's pre-fix
ROCm numbers. It sits between *Key numbers* and *Reproduction* on purpose — it
is a second host, not a re-measurement of the A6000 matrix above.

**Host/env:** AMD Instinct MI300X (one VF, single tenant), ROCm 7.1, venv
`/tmp/genesis/task_3858dfe003526650/etlbench-venv/bin/python` (Python 3.14,
torch 2.13.0+rocm7.1), etl 0.1.0, `evox_etl` imported from this repo's
`src/`. Seed 42, 100 generations (caps noted), 2 warmup steps excluded, and
`ms/step = 1000 · run_time_s / gens` — same case tables, metrics, seeds and
timing semantics as the CUDA sections (the `_mi300x` suffix marks the
CPU-interpreter column re-run on this host, so the A6000
`{so,mo}_etl-iree-llvm-cpu.json` files are never overwritten). torch's device
string is `"cuda"` on ROCm too — HIP is exposed through `torch.cuda`, there is
no `"rocm"` torch device.

**Cell format in this section:** `pre-fix → post-fix / speedup×`, where
`speedup = torch-rocm ms/step ÷ etl ms/step` (**>1.0 = etl faster than
torch-rocm**, **<1.0 = etl slower**). `err` = error record, `@Ng` = gens cap,
`-` = ratio undefined because one side is `err`, `1.00×` = tie inside
measurement noise, `etl-numpy (reused)` = baseline value only (not re-run:
no old→new, no ratio).

### What changed

Four `evox_etl` fixes landed between the two matrices:

1. **batched `eigh` by default** — covariance eigendecompositions are issued
   batched instead of as a per-matrix loop;
2. **two-stage split reduction for rank-0 reductions** — scalar (rank-0)
   reduction results are produced by a split two-stage reduction;
3. **XLA per-call hot path** — the per-step xla execution call path was put on
   a hot path (every `step()` goes through it);
4. **emitter correctness fixes** — bool/int reductions and NaN-aware
   `argmin`/`argmax` now emit correct/exportable code.

A fifth change also moves these numbers: `src/evox_etl/core/workflow.py` now
**pipelines the monitor's device→host readback**, worth ~1.18× per step on
device-resident backends (the per-generation monitor state is read back while
the next step runs, instead of serializing with it).

**The old→new deltas below mix all five effects plus measurement noise and
cannot be attributed to any single fix.** The noise floor is visible in the
`torch-rocm` column itself, which is code-identical between the two rounds:
its SO cells move by −22.9% to +43.1% and its MO cells by +2.3% to +11.1% (see
*Measurement caveats*). Only the CMA-ES family (3.9–69×) and the device-resident
non-CMA-ES xla cells (1.23–1.60×) clearly exceed that floor.

### Provenance

The pre-fix ROCm matrix was **never committed to this repository before this
run**: the `torch-rocm` / `etl-xla-rocm` / `etl-iree-rocm` ids only became
runnable once the harness gained ROCm support (`bench_common.set_rocm_env`,
`ROCM_BACKENDS`, the ROCm PJRT-plugin recipe), and the previous round's ROCm
numbers existed only as scratch table dumps in that round's `$TMPDIR`. Every
"old" number in this section therefore comes from those scratch measurements,
re-committed as `results/rocm_prefix_baseline.json`: its `provenance` block
names the three source dumps, the host, the date, the pre-fix `evox_etl`
build and the reconstruction rules, and its `so` / `mo` maps hold
`ms_per_step` + `best_fitness` (`mo`: `hv`/`igd`/`n_pf_points`) per case and
backend — `ms_per_step: null` + `error: true` is an error record, a `gens`
key is a gens cap. It is a reference file, not a runner output: it carries no
`parity` blocks.

`etl-numpy` was **not re-run** in this round — the fixes target the compiled
device paths and the numpy interpreter is unaffected — so its numbers are
reused verbatim from that baseline file and are labelled `etl-numpy (reused)`
in every table below: no old→new arrow, no ratio. They mean "no change
claimed", not "post-fix measurement". That baseline records no gens caps for
`etl-numpy` (only the pre-fix `etl-iree-llvm-cpu` CMA-ES 10000x100 cells are
capped, at 22 gens), so the reused column carries plain values.

### Single-objective (SO) — PSO / DE / CMA-ES / OpenES

PS/PR/PA = PSO on Sphere/Rastrigin/Ackley, DS/DR/DA = DE, CS/CR/CA = CMA-ES,
OS/OR/OA = OpenES.

#### 100 × 10

| backend | PS | PR | PA | DS | DR | DA | CS | CR | CA | OS | OR | OA |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `torch-rocm` | 0.531 → 0.477 | 0.527 → 0.5436 | 0.588 → 0.5878 | 0.319 → 0.3858 | 0.36 → 0.3904 | 0.444 → 0.5604 | err → err | err → err | err → err | 0.203 → 0.2134 | 0.246 → 0.2505 | 0.306 → 0.2947 |
| `etl-xla-rocm` | 0.544 → 0.4277 / 1.12× | 0.577 → 0.4228 / 1.29× | 0.557 → 0.4186 / 1.40× | 0.59 → 0.3899 / 0.99× | 0.486 → 0.3756 / 1.04× | 0.492 → 0.4016 / 1.40× | 7.54 → 1.582 / - | 7.62 → 1.598 / - | 7.41 → 1.64 / - | 0.516 → 0.3934 / 0.54× | 0.574 → 0.4054 / 0.62× | 0.517 → 0.3876 / 0.76× |
| `etl-iree-rocm` | 0.456 → 0.4034 / 1.18× | 0.443 → 0.4533 / 1.20× | 0.45 → 0.4087 / 1.44× | 0.39 → 0.3524 / 1.09× | 0.403 → 0.3753 / 1.04× | 0.402 → 0.3615 / 1.55× | 21.9 → 5.481 / - | 21.6 → 5.076 / - | 22.2 → 5.052 / - | 0.41 → 0.3805 / 0.56× | 0.425 → 0.3755 / 0.67× | 0.544 → 0.3763 / 0.78× |
| `etl-iree-llvm-cpu` | 0.204 → 0.2198 / 2.17× | 0.263 → 0.2625 / 2.07× | 0.273 → 0.2781 / 2.11× | 0.213 → 0.2057 / 1.88× | 0.233 → 0.2302 / 1.70× | 0.242 → 0.2387 / 2.35× | 9.78 → 2.402 / - | 9.7 → 1.698 / - | 10.5 → 2.667 / - | 0.228 → 0.2398 / 0.89× | 0.257 → 0.2579 / 0.97× | 0.264 → 0.266 / 1.11× |
| `etl-numpy` (reused) | 0.534 | 0.575 | 0.609 | 0.611 | 0.649 | 0.679 | 0.884 | 0.995 | 1.05 | 0.419 | 0.457 | 0.504 |

#### 1000 × 50

| backend | PS | PR | PA | DS | DR | DA | CS | CR | CA | OS | OR | OA |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `torch-rocm` | 0.504 → 0.4835 | 0.532 → 0.4368 | 0.439 → 0.6004 | 0.324 → 0.3403 | 0.491 → 0.4027 | 0.398 → 0.5696 | err → err | err → err | err → err | 0.201 → 0.2133 | 0.257 → 0.2555 | 0.276 → 0.2915 |
| `etl-xla-rocm` | 0.604 → 0.4293 / 1.13× | 0.623 → 0.4195 / 1.04× | 0.612 → 0.4261 / 1.41× | 0.516 → 0.389 / 0.87× | 0.524 → 0.3889 / 1.04× | 0.533 → 0.4114 / 1.38× | 262 → 8.307 / - | 266 → 8.64 / - | 260 → 8.768 / - | 0.535 → 0.3998 / 0.53× | 0.569 → 0.398 / 0.64× | 0.545 → 0.4029 / 0.72× |
| `etl-iree-rocm` | 0.648 → 0.6331 / 0.76× | 0.669 → 0.6389 / 0.68× | 0.66 → 0.6252 / 0.96× | 0.616 → 0.5601 / 0.61× | 0.634 → 0.5765 / 0.70× | 0.615 → 0.5527 / 1.03× | 882 → 40.39 / - | 902 → 40.29 / - | 925 → 40.15 / - | 0.718 → 0.6525 / 0.33× | 0.717 → 0.6572 / 0.39× | 0.711 → 0.6694 / 0.44× |
| `etl-iree-llvm-cpu` | 1.1 → 1.239 / 0.39× | 1.79 → 1.912 / 0.23× | 1.89 → 2.025 / 0.30× | 0.993 → 1.058 / 0.32× | 1.68 → 1.767 / 0.23× | 1.79 → 1.88 / 0.30× | 661 → 34.11 / - | 659 → 34.96 / - | 664 → 35.98 / - | 1.39 → 1.535 / 0.14× | 2.1 → 2.187 / 0.12× | 2.14 → 2.167 / 0.13× |
| `etl-numpy` (reused) | 1.4 | 1.51 | 1.56 | 1.29 | 1.44 | 1.48 | 2.23 | 2.3 | 2.35 | 0.745 | 0.826 | 0.867 |

#### 10000 × 100

| backend | PS | PR | PA | DS | DR | DA | CS | CR | CA | OS | OR | OA |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `torch-rocm` | 0.574 → 0.5795 | 0.607 → 0.6361 | 0.501 → 0.6863 | 0.365 → 0.3967 | 0.574 → 0.4428 | 0.437 → 0.472 | err → err | err → err | err → err | 0.242 → 0.2583 | 0.289 → 0.3047 | 0.315 → 0.3251 |
| `etl-xla-rocm` | 0.612 → 0.4707 / 1.23× | 0.62 → 0.43 / 1.48× | 0.701 → 0.4393 / 1.56× | 0.514 → 0.396 / 1.00× | 0.558 → 0.3773 / 1.17× | 0.506 → 0.3758 / 1.26× | 1256 → 18.22 / - | 1220 → 18.53 / - | 1246 → 19.35 / - | 0.603 → 0.416 / 0.62× | 0.572 → 0.4108 / 0.74× | 0.616 → 0.4275 / 0.76× |
| `etl-iree-rocm` | 2.61 → 0.5511 / 1.05× | 2.63 → 0.571 / 1.11× | 2.63 → 0.5769 / 1.19× | 2.28 → 0.4651 / 0.85× | 2.29 → 0.4829 / 0.92× | 2.29 → 0.4878 / 0.97× | 4127 → 94.59 / - | 4107 → 94.47 / - | 4095 → 94.64 / - | 3.16 → 0.6561 / 0.39× | 3.13 → 0.6739 / 0.45× | 3.14 → 0.6756 / 0.48× |
| `etl-iree-llvm-cpu` | 13.4 → 14.72 / 0.04× | 25.6 → 27.01 / 0.02× | 27.4 → 28.11 / 0.02× | 14.1 → 14.25 / 0.03× | 27.7 → 29.2 / 0.02× | 26.9 → 27.78 / 0.02× | 10578@22g → 2182 / - | 10515@22g → 2152 / - | 10607@22g → 2186 / - | 21.6 → 22.11 / 0.01× | 34.3 → 35.18 / 0.01× | 35 → 36.38 / 0.01× |
| `etl-numpy` (reused) | 27.8 | 34.1 | 32.9 | 18.9 | 18.7 | 20.1 | 29.3 | 31.6 | 28.4 | 11.3 | 20.6 | 17.4 |

**CMA-ES is where the fixes are largest.** At 10000x100 the SO CMA-ES cells go
from 1256 → **18.22** ms/step (`etl-xla-rocm`, 68.9×), 4127 → **94.59**
(`etl-iree-rocm`, 43.6×) and 10578@22g → **2182** (`etl-iree-llvm-cpu`,
4.85×); at 1000x50 from 262 → 8.31 (31.5×) / 882 → 40.4 (21.8×) / 661 → 34.1
(19.4×); at 100x10 from 7.54 → 1.58 (4.8×) / 21.9 → 5.48 (4.0×) / 9.78 → 2.40
(4.1×). **The pre-fix 22-gen cap on the llvm-cpu 10000x100 CMA-ES cells is no
longer needed**: per-step cost fell 4.85–4.89×, so the full 100-generation run
(215–219 s) now costs slightly *less* than the old 22-generation capped run
(231–233 s), and all three compiled etl backends run the full 36-cell SO
matrix with **no gens caps and no error records**. Over the 27 non-CMA-ES SO
cells the old→new improvement is 1.23–1.60× (median 1.35×) on xla, 0.98–4.90×
(median 1.11×) on iree and 0.89–1.04× (median 0.96×) on llvm-cpu — the device
backends moved, the CPU interpreter (which these fixes do not target) did not.

### Multi-objective (MO) — NSGA2 / NSGA3 / MOEAD

D1/D2 = DTLZ1/DTLZ2. `hv`/`igd` for every cell are in the JSONs; the timings
are the comparison here. MOEAD's effective population is the Das-Dennis vector
count on both sides.

#### 100 × 3 × 10

| backend | NSGA2/1 | NSGA2/2 | NSGA3/1 | NSGA3/2 | MOEAD/1 | MOEAD/2 |
|---|---|---|---|---|---|---|
| `torch-rocm` | 3.07 → 3.318 | 2.71 → 2.822 | 176 → 185.3 | 243 → 254.3 | 159 → 166.8 | 168 → 172.0 |
| `etl-xla-rocm` | 1.02 → 0.8361 / 3.97× | 0.84 → 0.6413 / 4.40× | 3.02 → 2.349 / 78.91× | 2.6 → 1.984 / 128.20× | 5.24 → 5.309 / 31.42× | 5.29 → 5.573 / 30.87× |
| `etl-iree-rocm` | err → err / - | err → err / - | err → err / - | err → err / - | err → err / - | err → err / - |
| `etl-iree-llvm-cpu` | err → err / - | err → err / - | err → err / - | err → err / - | err → 4.673 / 35.70× | err → err / - |
| `etl-numpy` (reused) | 4.82 | 4.13 | 7.21 | 5.5 | 32.7 | 33.1 |

#### 1000 × 3 × 30

| backend | NSGA2/1 | NSGA2/2 | NSGA3/1 | NSGA3/2 | MOEAD/1 | MOEAD/2 |
|---|---|---|---|---|---|---|
| `torch-rocm` | 4.29 → 4.446 | 3.08 → 3.165 | 253 → 268.7 | 249 → 276.6 | 1743 → 1803 | 1840 → 1882 |
| `etl-xla-rocm` | 1.56 → 1.282 / 3.47× | 1.04 → 0.8066 / 3.92× | 3.55 → 2.844 / 94.50× | 3.12 → 2.279 / 121.37× | 48.2 → 52.89 / 34.09× | 52.4 → 53.77 / 34.99× |
| `etl-iree-rocm` | err → err / - | err → err / - | err → err / - | err → err / - | err → err / - | err → err / - |
| `etl-iree-llvm-cpu` | err → err / - | err → err / - | err → err / - | err → err / - | err → 134.1 / 13.45× | err → 130.0 / 14.47× |
| `etl-numpy` (reused) | 201 | 164 | 257 | 226 | 383 | 387 |

- **`etl-xla-rocm` is a full 12/12 MO matrix** and beats `torch-rocm` on every
  cell: NSGA2 3.47–4.40×, NSGA3 78.9–128.2×, MOEAD 30.9–35.0× (median over
  the 12 cells 32.76×). Old→new: the 8 NSGA2/NSGA3 cells improved 1.22–1.37×, the 4 MOEAD
  cells are 0.91–0.99× — flat within the run-to-run spread visible in the
  torch column, so no MOEAD improvement is claimed.
- **`etl-iree-rocm` MO is `err` on all 12 cells** (documented upstream
  iree-compile SIGSEGV on the `rocm` target; see *Measurement caveats*), so
  every ratio is `-`.
- **`etl-iree-llvm-cpu` MO runs 3 of 12 cells** —
  `MOEAD/DTLZ1/100x3x10` 4.67 ms/step (35.70×), `MOEAD/DTLZ1/1000x3x30` 134.1
  (13.45×), `MOEAD/DTLZ2/1000x3x30` 130.0 (14.47×) — and is `err` on the other
  9 (compile failures). This is 3/12 where the pre-fix round had 0/12.

### Still slower than torch-rocm (ratio < 1.0)

Comparable cells only (both sides non-`err`; the 9 CMA-ES SO cells are never
comparable — no torch reference). **No MO cell is in this list.**

- **`etl-xla-rocm`: 11 of 27 cells** — all nine OpenES cells (0.53–0.76×:
  `OS` 0.54/0.53/0.62, `OR` 0.62/0.64/0.74, `OA` 0.76/0.72/0.76 at
  100x10/1000x50/10000x100) plus `DE/Sphere/100x10` 0.99× and
  `DE/Sphere/1000x50` 0.87×. The other 16 are faster (up to 1.56×); median
  1.04×.
- **`etl-iree-rocm`: 17 of 27 cells** — all nine OpenES cells (0.33–0.78×),
  the three PSO 1000x50 cells (`PS` 0.76, `PR` 0.68, `PA` 0.96) and five DE
  cells (`DS` 0.61 @1000x50 / 0.85 @10000x100, `DR` 0.70 / 0.92, `DA` 0.97
  @10000x100 — `DE/Ackley/1000x50` is a 1.03× win). Ten cells are faster (up
  to 1.55×); median 0.85×.
- **`etl-iree-llvm-cpu`: 20 of 27 cells** — every 1000x50 and 10000x100 cell
  (0.01–0.39×, worst at 10000x100) plus `OpenES/Sphere/100x10` 0.89× and
  `OpenES/Rastrigin/100x10` 0.97×. Only seven cells win (1.11–2.35×, all at
  100x10); median 0.23×.
- **MO: none.** `etl-xla-rocm` 12/12 faster (3.47–128.20×, median 32.76×);
  `etl-iree-llvm-cpu` 3/3 comparable cells faster (13.45–35.70×);
  `etl-iree-rocm` has no comparable MO cell.

Hypotheses, and what I consider most likely per group:

1. **A constant ~0.35–0.45 ms/step host-side floor on the compiled etl
   backends** (execute call + monitor readback + Python dispatch) that
   dominates cheap per-step device work. The evidence is scale invariance:
   across all 27 non-CMA-ES SO cells `etl-xla-rocm` never leaves
   **0.376–0.471 ms/step** while the population grows 100× (100x10 →
   10000x100), whereas `torch-rocm` spans 0.213–0.686 over the same cells.
   OpenES isolates it because its device work is tiny: etl xla sits at
   0.388–0.427 and etl iree at 0.375–0.676 ms/step nearly flat across all
   three scales, while torch eager scales 0.213 → 0.325. Where torch's own
   per-step overhead is already large (PSO, and DE at 10000x100) the same
   floor is hidden and etl wins. **Most likely for the 11 xla cells and the
   OpenES part of the 17 iree cells**; iree's floor is higher than xla's
   (0.47–0.68 ms/step at ≥1000x50 vs xla's 0.38–0.47), which is exactly why
   iree loses the 1000x50 PSO/DE cells that xla ties or wins.
2. **CMA-ES on iree still pays the cyclic-Jacobi `eigh` lowering** — iree
   ships no native `stablehlo.eigh`, so the per-step covariance eigh lowers to
   an O(n⁴) while-loop composition. The old→new drop (4127 → 94.6 ms/step at
   dim 100, and 4.0–43.6× across the family) reflects the batched-eigh default
   together with the other fixes — but it did **not** give iree a native eigh:
   iree is still 3.1–5.2× slower than xla on identical CMA-ES cells, and the
   gap widens with dimension (iree/xla 3.08–3.46× at dim 10, 4.58–4.86× at
   dim 50, 4.89–5.19× at dim 100), the signature of a super-linear lowering
   rather than a fixed overhead. These cells are absent from the list above
   only because `torch-rocm` CMA-ES is `err`. **Most likely for iree's absolute
   CMA-ES cost, the largest single-backend deficit left in the matrix.**
3. **`etl-iree-llvm-cpu` is a CPU interpreter, not a GPU path** — at 10000x100
   (torch 0.40–0.69 ms/step) it loses by 1–2 orders (0.009–0.04×) and at
   1000x50 by 2.6–8.5× (0.12–0.39×). It wins only at 100x10, where
   torch-rocm's per-step cost is pure overhead (0.39–0.59 ms/step on the
   PSO/DE cells) and the interpreter runs at 0.21–0.28 ms/step; it still loses
   the two OpenES 100x10 cells (0.89×/0.97×), where torch-rocm is down at
   0.21–0.29 ms/step. **Most likely for all 20 llvm-cpu cells.**
4. **In-graph monitor topk/gather as the marginal cost at 10000x100** — the
   *marginal* device cost is bounded by measurement, not separated: growing
   the population 100× moves the xla floor by only ~0.02–0.06 ms/step
   (0.376–0.428 at 100x10 → 0.376–0.471 at 10000x100), so all scale-dependent
   device work at that size is of that order. Plausible as the marginal
   component of hypothesis 1; **not separately provable from this matrix** (it
   needs a monitor-ablation run) and not the driver of any losing cell on its
   own.

### Measurement caveats

- **Two `torch-rocm` MO sweeps.** The committed `torch-rocm` MO column is the
  second of two full sweeps this round; the first differed by −0.6% to +7.1%
  per cell (slower on 10 of the 12 cells). A separate two-case spot run of
  `NSGA3/DTLZ2/100x3x10` measured 186.2 ms/step against 254.3–272.3 in the two
  full sweeps, so NSGA3's run-to-run spread is ~±20% while NSGA2/MOEAD stay
  within ~7%. NSGA3 is additionally known not to
  reproduce point-for-point across GPU runs (its reference-point machinery is
  data- and order-dependent), so NSGA3 cells should be read as
  "same order of magnitude", not as exact.
- **The torch path is unchanged by the fixes.** `torch-rocm` old→new
  differences are pure measurement variance: SO spans −22.9% to +43.1%
  (`DE/Rastrigin/10000x100` 0.574 → 0.443, `DE/Ackley/1000x50` 0.398 → 0.570)
  and MO +2.3% to +11.1%. Single-cell old→new deltas of that magnitude are
  noise, not code effects — which is why only the CMA-ES family and the
  device-resident non-CMA-ES xla cells (1.23–1.60×) are claimed as
  improvements.
- **The torch CMA-ES reference does not exist on this host.** All 9
  `torch-rocm` CMA-ES SO records are `err` with `UncapturedHigherOrderOpError:
  ... torch.cond` (a torch 2.13 incompatibility that also reproduces on
  torch-cpu). Their ratio cells are `-` by construction, not "etl lost".
- **`etl-iree-rocm` MO: 12/12 `err`** — the documented upstream iree-compile
  SIGSEGV (`BackendError: iree-compile failed to compile ... Error code: -11`,
  stack inside `libIREECompiler.so`) on the `rocm` target (`--iree-rocm-target=gfx942`),
  failing on every compile attempt — each case carries the
  `documented: iree-compile segfault (SIGSEGV, error -11) compiling the MO
  program for the rocm target; deterministic across 4 in-process and 4
  fresh-process attempts` note. No
  ratio, no parity, nothing to attribute to evox_etl.
- **`etl-iree-llvm-cpu` MO: 3/12** — MO compilation on this backend is
  nondeterministic, so the three working cells needed process-level retries
  (the `ETL_COMPILE_ATTEMPTS` recipe); the other 9 are `err`.
- **`torch-rocm` can abort in HIP teardown** after the JSON is written
  (`c10::cuda::SetDevice` inside a `TensorImpl` destructor). The JSON/stdout is
  the source of truth; that abort is not a failed run and cannot affect
  timings (they are taken long before).
- **One VF, single tenant:** runs were serial and foreground; no co-tenant and
  no concurrent ROCm job, so no cross-run interference is expected beyond the
  sweep-to-sweep spread above.

### Parity summary

Parity follows the existing convention — each record vs the **`torch-cpu`**
baseline (`results/{so,mo}_torch-cpu.json`, same `case_id`): relative error
≤ 10% → ok, both sides ≤ 1e-4 → ok ("both converged to machine zero"),
error/gens mismatch → skip. Counts over the 192 new records:

| backend | suite | ok | not-ok | skip | machine-zero oks |
|---|---|---|---|---|---|
| `torch-rocm` | SO | 15 | 12 | 9 | 1 |
| `etl-xla-rocm` | SO | 21 | 15 | 0 | 2 |
| `etl-iree-rocm` | SO | 21 | 15 | 0 | 2 |
| `etl-iree-llvm-cpu` | SO | 23 | 13 | 0 | 2 |
| **SO total** | | **80** | **55** | **9** | 7 |
| `torch-rocm` | MO | 6 | 6 | 0 | 0 |
| `etl-xla-rocm` | MO | 3 | 9 | 0 | 0 |
| `etl-iree-rocm` | MO | 0 | 0 | 12 | 0 |
| `etl-iree-llvm-cpu` | MO | 0 | 3 | 9 | 0 |
| **MO total** | | **9** | **18** | **21** | 0 |

- **Every not-ok cell is `RNG-stream variance on unconverged run`** — no new
  note text, no correctness verdict change. The largest are
  `MOEAD/DTLZ1/1000x3x30` rel 21.01 (xla and llvm-cpu),
  `NSGA3/DTLZ1/1000x3x30` rel 3.84 (xla), `CMAES/Ackley/100x10` rel ≈ 5.6e3
  (all three etl backends — Ackley's flat-plateau convergence scatter) plus
  the small-scale DE/OpenES/Rastrigin cells. MO not-ok is concentrated in
  DTLZ1 at 100 gens (both sides still far from the front) and in DTLZ2's igd.
- **All skips are error records:** 9 `torch-rocm` CMA-ES SO cells
  (`torch.cond`), 12 `etl-iree-rocm` MO cells (compile SIGSEGV), 9
  `etl-iree-llvm-cpu` MO cells (compile failures). Error records carry
  `parity.ok = null` with `skip: "backend error"`.
- **No new record is gens-capped** — all 192 ran the full 100 gens (the
  pre-fix 22-gen llvm-cpu CMA-ES cap is gone). The reused `etl-numpy` column
  has no post-fix parity verdicts: those values are copied from
  `results/rocm_prefix_baseline.json`, which records no gens cap for them.
- The three compiled etl backends share a 13-cell SO not-ok core (xla and iree
  add `CMAES/Rastrigin/1000x50` and `/10000x100`) but are **not** bit-identical
  on this box, unlike the CUDA round: `PSO/Sphere/100x10` best = 2.44e-08
  (xla) vs 3.31e-08 (iree); `CMAES/Rastrigin/100x10` best = 2.59 (xla) /
  1.13 (iree) / 14.41 (llvm-cpu). Reduction order and RNG-stream differences
  on hardware targets — consistent with the not-ok notes, not a defect signal.

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

- `results/{so,mo}_{backend}.json` — 12 A6000-host files, 288 records with
  parity verdicts vs torch-cpu written in.
- `results/{so,mo}_{torch-rocm,etl-xla-rocm,etl-iree-rocm}.json` and
  `results/{so,mo}_etl-iree-llvm-cpu_mi300x.json` — the 8 MI300X/ROCm
  measurement files (192 records with parity verdicts vs torch-cpu). The
  `_mi300x` suffix keeps the A6000 `etl-iree-llvm-cpu` files intact.
- `results/rocm_prefix_baseline.json` — pre-fix ROCm reference matrix
  (`provenance` block + `so`/`mo` maps; no parity blocks, not a runner
  output) that every "old" number in the ROCm section traces back to.
- `retag_parity.py` — partial-run merge + parity verdict re-derivation.
- `style_comparison.md` — code style/cleanliness comparison (torch OOP vs
  etl functional).
- `bench_common.py` — shared case tables, metric implementations, parity
  helpers (pure stdlib + numpy).
- `bench_so.py` / `bench_mo.py` / `run.py` — the runners and CLI dispatcher.
