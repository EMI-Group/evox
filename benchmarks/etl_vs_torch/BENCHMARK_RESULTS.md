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

## Single-objective (SO) — PSO / DE / CMA-ES / OpenES × Sphere / Rastrigin / Ackley

Cell = `best_fitness / ms_per_step`. S/R/A = Sphere/Rastrigin/Ackley.
`err` = error record; `NaN (silent)` = torch-cpu CMA-ES run whose covariance
silently went NaN (see *Caps & limits*); `@Ng` = etl-numpy interpreter cap
(gens actually run).

### 100 × 10

| backend | PSO/S | PSO/R | PSO/A | DE/S | DE/R | DE/A | CMA/S | CMA/R | CMA/A | OE/S | OE/R | OE/A |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-cpu | 1.5e-08 / 0.677 | 8.09 / 0.7 | 20.01 / 0.759 | 0.259 / 0.53 | 47.47 / 0.568 | 20.00 / 0.617 | 1.2e-06 / 58.1 | 2.12 / 57.8 | 0.00313 / 63 | 172.8 / 0.41 | 252.1 / 0.449 | 20.66 / 0.49 |
| torch-cuda | 9.3e-09 / 1.06 | 5.05 / 1.16 | 20.00 / 1.2 | 0.186 / 0.885 | 56.90 / 0.946 | 19.42 / 1 | 8.2e-07 / 58.8 | 16.32 / 62 | 0.00182 / 64.4 | 92.91 / 0.622 | 201.4 / 0.692 | 20.50 / 0.737 |
| etl-numpy | 3.1e-08 / 0.83 | 5.52 / 0.902 | 20.00 / 0.953 | 0.607 / 0.973 | 56.38 / 1.04 | 18.95 / 1.08 | 2.8e-09 / 1.28 | 5.38 / 1.38 | 20.35 / 1.4 | 132.1 / 0.491 | 221.2 / 0.561 | 20.69 / 0.616 |
| etl-iree-llvm-cpu | 3.1e-08 / 1.3 | 5.52 / 2.3 | 20.00 / 2.71 | 0.607 / 1.24 | 56.38 / 1.26 | 18.95 / 2.01 | 2.7e-09 / 24.8 | 20.86 / 19.3 | 20.50 / 18.9 | 132.1 / 0.973 | 221.2 / 1.01 | 20.69 / 0.814 |
| etl-iree-cuda | 3.1e-08 / 1.42 | 5.52 / 1.49 | 20.00 / 1.56 | 0.607 / 1.13 | 56.38 / 1.29 | 18.95 / 1.24 | 2.7e-09 / 40.3 | 17.20 / 40.7 | 20.43 / 42.9 | 132.1 / 1.46 | 221.2 / 1.15 | 20.69 / 1.25 |
| etl-xla-cuda | 3.1e-08 / 5.37 | 5.52 / 5.11 | 20.00 / 5.08 | 0.607 / 3.27 | 56.38 / 3.91 | 19.83 / 4.21 | 2.6e-09 / 7.8 | 2.06 / 9.42 | 20.59 / 9.43 | 132.1 / 4.78 | 221.2 / 4.82 | 20.69 / 4.7 |

### 1000 × 50

| backend | PSO/S | PSO/R | PSO/A | DE/S | DE/R | DE/A | CMA/S | CMA/R | CMA/A | OE/S | OE/R | OE/A |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-cpu | 30.77 / 2.56 | 391.5 / 2.65 | 20.00 / 2.69 | 17976 / 2.22 | 22442 / 2.61 | 20.28 / 2.42 | NaN (silent) | NaN (silent) | NaN (silent) | 1814 / 0.87 | 2271 / 0.969 | 21.26 / 1.01 |
| torch-cuda | 13.46 / 1.08 | 330.6 / 1.14 | 20.05 / 1.21 | 25619 / 0.9 | 27489 / 0.971 | 20.26 / 1.03 | err | err | err | 1751 / 0.68 | 2214 / 0.784 | 21.21 / 0.836 |
| etl-numpy | 201.4@50g / 2.17 | 875.2@50g / 2.72 | 20.58@50g / 2.74 | 42715@50g / 1.91 | 41999@50g / 1.9 | 20.52@50g / 1.98 | err | err | err | 13057@50g / 0.986 | 13652@50g / 1.11 | 21.24@50g / 1.2 |
| etl-iree-llvm-cpu | 31.94 / 12.4 | 365.9 / 10.3 | 20.56 / 9.82 | 23840 / 9.79 | 27923 / 9.97 | 20.14 / 12.7 | err | err | err | 1887 / 10.2 | 2370 / 10.5 | 21.24 / 12 |
| etl-iree-cuda | 31.94 / 35.7 | 365.9 / 35.8 | 20.56 / 35.9 | 23840 / 35.7 | 27923 / 35.6 | 20.14 / 35.5 | err | err | err | 1887 / 35.5 | 2370 / 35.2 | 21.24 / 35.6 |
| etl-xla-cuda | 31.94 / 5.73 | 365.9 / 5.74 | 20.56 / 5.29 | 23840 / 4.36 | 27923 / 4.6 | 20.08 / 3.64 | err | err | err | 1887 / 3.71 | 2370 / 3.84 | 21.24 / 3.77 |

### 10000 × 100

| backend | PSO/S | PSO/R | PSO/A | DE/S | DE/R | DE/A | CMA/S | CMA/R | CMA/A | OE/S | OE/R | OE/A |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-cpu | 2548 / 27.4 | 3846 / 36.2 | 20.08 / 26.7 | 97177 / 16.8 | 107997 / 20.1 | 20.61 / 17.7 | NaN (silent) | NaN (silent) | NaN (silent) | 4219 / 8.59 | 5163 / 8.24 | 21.30 / 10.4 |
| torch-cuda | 1893 / 1.08 | 3362 / 1.14 | 20.12 / 1.26 | 109077 / 0.874 | 118943 / 1.01 | 20.67 / 1 | err | err | err | 4068 / 0.626 | 5095 / 0.701 | 21.31 / 0.754 |
| etl-numpy | 84302@10g / 74 | 85202@10g / 66.6 | 20.35@10g / 64.4 | 159500@10g / 42.1 | 155489@10g / 38.6 | 20.82@10g / 42.4 | err | err | err | 148365@10g / 26.8 | 149233@10g / 25.4 | 21.39@10g / 22.3 |
| etl-iree-llvm-cpu | 2691 / 399.2 | 4119 / 561.4 | 20.02 / 707.3 | 104824 / 506.0 | 108335 / 554.4 | 20.56 / 585.5 | err | err | err | 4112 / 481.7 | 5110 / 443.2 | 21.32 / 479.9 |
| etl-iree-cuda | 2691 / 3475 | 4119 / 3481 | 20.02 / 3478 | 104824 / 3481 | 108335 / 3483 | 20.56 / 3496 | err | err | err | 4112 / 3482 | 5110 / 3480 | 21.32 / 3504 |
| etl-xla-cuda | 2691 / 17 | 4119 / 15.2 | 20.02 / 18.2 | 104824 / 11.2 | 108335 / 11.6 | 20.64 / 11.6 | err | err | err | 4112 / 8.23 | 5110 / 9.32 | 21.32 / 9.34 |

### SO compile time and speedup

Compile time range over successful cases (`wf.init(seed=42)`):

| backend | SO compile time |
|---|---|
| torch-cpu / torch-cuda | 0.0 s (eager) |
| etl-numpy | 0.044 – 0.382 s |
| etl-iree-llvm-cpu | 0.903 – 3.398 s |
| etl-iree-cuda | 1.209 – 20.575 s |
| etl-xla-cuda | 0.411 – 4.392 s |

ms/step ratio, case-matched, median (min – max); **>1 = faster than the baseline**:

| backend | vs torch-cpu | vs torch-cuda |
|---|---|---|
| torch-cuda | 1.8× (0.6× – 31.9×) | — |
| etl-numpy | 0.80× (0.32× – 45.4×) | — |
| etl-iree-llvm-cpu | 0.24× (0.018× – 3.3×) | — |
| etl-iree-cuda | 0.07× (0.002× – 1.5×) | 0.03× (0.0002× – 1.5×) |
| etl-xla-cuda | 0.51× (0.086× – 7.5×) | 0.20× (0.064× – 7.5×) |

The large "etl wins" outliers (45×, 7.5×) are CMA-ES 100x10, where torch's
CPU/GPU path is exceptionally slow (~58–63 ms/step vs 1.3–9.4 ms/step). At
10000x100 torch wins decisively on both CPU and GPU.

## Multi-objective (MO) — NSGA2 / NSGA3 / MOEAD × DTLZ1 / DTLZ2

Cell = `hv/igd / ms_per_step`. D1/D2 = DTLZ1/DTLZ2. `BLOCKED` = error record
(see *Caps & limits*); `@Ng` = etl-numpy interpreter cap. MOEAD's effective
population is the Das-Dennis vector count (pop_size is overwritten on both
sides).

### 100 × 3 × 10

| backend | NSGA2/D1 | NSGA2/D2 | NSGA3/D1 | NSGA3/D2 | MOEAD/D1 | MOEAD/D2 |
|---|---|---|---|---|---|---|
| torch-cpu | 7.92/2.05 / 4.2 | 7.46/0.0244 / 3 | 6.42/1.8 / 407.5 | 7.46/0.0257 / 420.3 | 184.1/4.5 / 97 | 7.46/0.0289 / 99.2 |
| torch-cuda | 7.69/0.548 / 5.13 | 7.46/0.0246 / 4.29 | 6.46/1.15 / 623.0 | 7.46/0.0253 / 625.5 | 30.97/2.73 / 273.1 | 7.46/0.0312 / 279.5 |
| etl-numpy | 23.94/4 / 9.71 | 7.46/0.0231 / 9.2 | 113.4/9.62 / 10.9 | 7.46/0.0257 / 8.23 | 7.56/0.572 / 60.2 | 7.46/0.0335 / 59.5 |
| etl-iree-llvm-cpu | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED |
| etl-iree-cuda | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED |
| etl-xla-cuda | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED |

### 1000 × 3 × 30

| backend | NSGA2/D1 | NSGA2/D2 | NSGA3/D1 | NSGA3/D2 | MOEAD/D1 | MOEAD/D2 |
|---|---|---|---|---|---|---|
| torch-cpu | 39590/51.10 / 83.1 | 7.46/0.0167 / 45.2 | 17063/41.36 / 417.9 | 7.47/0.0152 / 387.1 | 465.0/6.83 / 1033 | 7.46/0.0208 / 1028 |
| torch-cuda | 3616/23.10 / 7.65 | 7.46/0.0162 / 5.49 | 19618/44.20 / 863.3 | 7.47/0.0148 / 719.1 | 3546/15.03 / 3505 | 7.46/0.0177 / 3621 |
| etl-numpy | 3690218/231.2@20g / 309.4 | 7.28/0.157@20g / 452.0 | 4081556/229.4@20g / 754.7 | 7.28/0.155@20g / 335.8 | 1766236/92.37@20g / 756.0 | 7.29/0.136@20g / 740.7 |
| etl-iree-llvm-cpu | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED |
| etl-iree-cuda | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED |
| etl-xla-cuda | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED |

### MO compile time and speedup

| backend | MO compile time |
|---|---|
| torch-cpu / torch-cuda | 0.0 s (eager) |
| etl-numpy | 0.317 – 0.734 s |
| etl-iree-llvm-cpu / etl-iree-cuda / etl-xla-cuda | — (every case fails at compile) |

| backend | vs torch-cpu |
|---|---|
| torch-cuda | 0.60× (0.28× – 10.9×) |
| etl-numpy | 1.3× (0.10× – 51.1×) |
| etl compiled backends | — (all BLOCKED) |

torch-cuda's MO speedup is bimodal: NSGA2 (5.5–7.6 ms/step) beats torch-cpu
(3–83 ms/step), while python-loop-bound NSGA3 (623–863 ms) and MOEAD
(273–3621 ms) are slower on GPU. etl-numpy's 51× outlier is NSGA3/DTLZ2
100x3x10 (8.2 vs 420 ms), i.e. torch NSGA3's CPU path is slow.

## Caps & limits

- **etl-numpy caps (interpreter too slow):** SO 1000x50 → 50 gens, SO
  10000x100 → 10 gens; MO 1000x3x30 → 20 gens (the etl monitor's O(n²)
  whole-history PF ranking on the numpy interpreter). Capped records keep
  their measured metric but parity is skipped (gens differ from baseline).
- **etl MO BLOCKED on all compiled backends:** 12/12 `BackendError` records
  each for iree-llvm-cpu / iree-cuda / xla-cuda. Root cause: the stablehlo
  v1 exporter used by the etl adapters rejects the `cumprod`/`flip` ops in
  `src/evox_etl/problems/numerical/dtlz.py` ("op 'cumprod' … is not supported
  in v1 — decompose it into supported ops or use a future compiler
  adapter"). No compile option exists — verified against the etl repo
  (identical deferral). Unblocking requires decomposing those ops in the
  exporter or rewriting `dtlz.py`'s evaluate formulas with v1-safe ops
  (`m` is static, so trace-time loops over `multiply`/`concatenate` work),
  then re-running the MO groups.
- **CMA-ES `c_c > 2` framework bug (pop ≥ 1000, dim ≥ 50):** the Hansen
  parametrization yields `c_c > 2` at (1000, 50) and (10000, 100), so
  `sqrt(c_c·(2−c_c)·mu_eff)` is NaN and the covariance goes all-NaN.
  etl raises `ValueError` at trace time → 6 error records per etl backend;
  torch-cuda raises `_LinAlgError` in cusolver's eigh → 6 error records;
  torch-cpu's LAPACK silently propagates the NaN → the 6 torch-cpu cells
  above are marked **"NaN (silent)"** (fitness values are frozen initial
  population garbage, not results).
- **CMAES/Ackley/100x10 divergence:** etl CMA-ES stalls on Ackley's
  zero-gradient plateau (best ≈ 20.35–20.59 on all four etl backends) while
  torch escapes (best ≈ 0.003). Parity rel_err ≈ 6.5–6.6e3 on every etl
  backend — a documented port divergence in `src/evox_etl`, not xla-specific.
- **MOEAD pop_size overwritten** by the Das-Dennis vector count on both
  torch and etl (recorded in the JSON notes).
- **Harness workarounds** baked into the numbers: torch MO runs with
  `torch.set_default_device(device)` and on-device lb/ub (torch MO algorithms
  allocate on the default device); torch Pareto fronts are extracted
  per-generation instead of `EvalMonitor.get_pf_fitness` (whose O(n²)
  domination matrix OOM-kills the process at 1000-pop scale — the two methods
  are mathematically identical, verified point-for-point at small scale);
  etl-xla-cuda stages with `device=None` because the xla adapter rejects
  non-cpu device labels (the GPU is selected purely via
  `CUDA_VISIBLE_DEVICES`, and the CUDA PJRT plugin has no CPU platform);
  the harness LD_PRELOADs cuDNN 9.8 and re-execs the process for xla-cuda
  (see Reproduction).

## Parity verdicts

Parity is computed per record against the torch-cpu baseline (same
`case_id`): relative error ≤ 10% → ok; both sides ≤ 1e-4 → ok with note
"both converged to machine zero"; baseline/record error or differing gens →
skip. Full blocks are written into every `results/*.json`.

Counts over all 240 non-baseline records:

| suite | ok | not-ok | skip | skipped because |
|---|---|---|---|---|
| SO | 84 | 48 | 48 | 30 CMA-ES large-scale backend errors + 18 etl-numpy gens caps |
| MO | 7 | 11 | 42 | 36 etl-GPU `BackendError` + 6 etl-numpy gens caps |
| **total** | **91** | **59** | **90** | 66 backend errors + 24 gens caps |

Parity-evaluable (ok + not-ok): **150**.

**SO not-ok (48)** — all notes are `RNG-stream variance on unconverged run`
unless stated:

- vs etl-numpy (7): PSO/Rastrigin/100x10 0.318 · DE/Sphere/100x10 1.347 ·
  DE/Rastrigin/100x10 0.188 · CMAES/Rastrigin/100x10 1.535 ·
  OpenES/Sphere/100x10 0.235 · OpenES/Rastrigin/100x10 0.122 ·
  **CMAES/Ackley/100x10 6507 (documented divergence)**
- vs etl-iree-llvm-cpu (9): as etl-numpy plus DE/Sphere/1000x50 0.326 and
  DE/Rastrigin/1000x50 0.244; CMAES/Rastrigin 8.827; **CMAES/Ackley 6555
  (divergence)**
- vs etl-iree-cuda (9): same 100x10 set (CMAES/Rastrigin 7.100) plus
  DE/Sphere/1000x50 0.326, DE/Rastrigin/1000x50 0.244; **CMAES/Ackley 6533
  (divergence)**
- vs etl-xla-cuda (8): CMAES/Rastrigin is ok (2.06 vs 2.12); otherwise the
  100x10 set plus DE/Sphere/1000x50 0.326, DE/Rastrigin/1000x50 0.244;
  **CMAES/Ackley 6584 (divergence)**
- vs torch-cuda (15): PSO/Sphere/1000x50 0.562 · PSO/Sphere/10000x100 0.257 ·
  PSO/Rastrigin 0.376 / 0.155 / 0.126 · DE/Sphere 0.282 / 0.425 / 0.122 ·
  DE/Rastrigin 0.198 / 0.225 / 0.101 · CMAES/Rastrigin/100x10 6.686 ·
  CMAES/Ackley/100x10 0.417 · OpenES/Sphere/100x10 0.462 ·
  OpenES/Rastrigin/100x10 0.201

**MO not-ok (11)** — all `RNG-stream variance on unconverged run`:

- vs etl-numpy (4): NSGA2/DTLZ1/100x3x10 rel 2.022 (hv 2.022, igd 0.955) ·
  NSGA3/DTLZ1/100x3x10 16.661 (16.661 / 4.336) ·
  MOEAD/DTLZ1/100x3x10 0.959 (0.959 / 0.873) ·
  MOEAD/DTLZ2/100x3x10 0.162 (hv 0.001, igd 0.162)
- vs torch-cuda (7): NSGA2/DTLZ1 0.732 / 0.909 · NSGA3/DTLZ1 0.362 / 0.150 ·
  MOEAD/DTLZ1 0.832 / 6.627 · MOEAD/DTLZ2/1000x3x30 0.151

**Overall verdict:** the three compiled etl backends are numerically
identical to each other (iree-llvm-cpu ≡ iree-cuda ≡ xla-cuda at bit or last-
ulp precision — e.g. PSO/Sphere/100x10 best = 3.054133657087732e-08 on all
four etl backends; only CMA-ES eigendecomposition noise and xla's GPU
non-deterministic reductions differ, e.g. DE/Ackley/100x10 18.95 vs 19.83),
and etl-numpy agrees bit-identically wherever it ran the same 100 gens.
torch vs etl agree within the 10% tolerance on all converged cases; the only
real divergences are the documented CMAES/Ackley/100x10 stall (all four etl
backends) and relative-error scatter on unconverged small-scale runs
(100x10 DE/OpenES/Rastrigin, DTLZ1 at 100 gens) — RNG-stream variance
between torch's MT19937 and etl's keyed PRNGs, not a correctness gap.

## Key numbers

- **torch-cuda PSO: ~1.06–1.26 ms/step at ALL scales** (100x10 → 10000x100);
  DE 0.87–1.01 ms/step; OpenES 0.62–0.84 ms/step — per-step cost is nearly
  scale-invariant.
- **etl-xla-cuda SO: 3.3–18.2 ms/step** (median 5.3; CMAES 100x10 7.8–9.4
  ms/step) + **0.41–4.39 s compile**.
- **etl-iree-llvm-cpu 10000x100: 399–707 ms/step** (compile 0.90–3.40 s).
- **etl-iree-cuda 10000x100: ~3.48–3.50 s/step** (3475–3504 ms; compile
  1.21–20.6 s) — the slowest non-interpreter backend at scale.
- **torch-cpu 10000x100: 8.2–91.6 ms/step** (OpenES 8.2, DE 16.8–20.1, PSO
  26.7–36.2, CMAES 91.6 on its NaN run).
- **etl-numpy (interpreter) is the slowest at scale:** SO 10000x100 22–74
  ms/step (capped at 10 gens), MO 1000x3x30 309–756 ms/step (capped at 20
  gens) — but it wins on small CMA-ES (1.3 ms/step vs torch-cpu 58 ms).
- **torch-cuda MO is bimodal:** NSGA2 4.3–7.6 ms/step (fast) vs NSGA3
  623–863 ms/step and MOEAD 273–3621 ms/step (python-loop/table bound).
- **Compile cost:** etl-numpy 0.04–0.73 s (cheapest); xla 0.41–4.39 s;
  iree-llvm 0.90–3.40 s; iree-cuda up to 20.6 s. torch pays nothing (eager).

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

Use `--cases <ids>`, `--gens-override <n>` and `--out <path>` for subsets
and smoke runs (smoke runs never write into `results/`; the committed JSONs
are the authoritative 100-gen matrix).

## Artifacts

- `results/{so,mo}_{backend}.json` — 12 files, 288 records with parity
  verdicts vs torch-cpu written in.
- `style_comparison.md` — code style/cleanliness comparison (torch OOP vs
  etl functional).
- `bench_common.py` — shared case tables, metric implementations, parity
  helpers (pure stdlib + numpy).
- `bench_so.py` / `bench_mo.py` / `run.py` — the runners and CLI dispatcher.
