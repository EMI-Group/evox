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
  MO full on torch-cpu, torch-cuda, etl-numpy (12 cases each, 100 gens);
  xla-cuda 8/12 (4 NSGA3 cells blocked); iree-llvm-cpu / iree-cuda 12/12
  error records.
- **etl-numpy caps:** SO 1000x50 → 50 gens, 10000x100 → 10 gens
  (interpreter too slow). The MO cap is removed: Pareto fronts are now
  extracted per-generation in `bench_mo.py` (`_etl_pf_fitness`, mirroring
  `_torch_pf_fitness`), replacing the O(n²) whole-history ranking that made
  100 gens infeasible — verified point-for-point identical.
- **etl-GPU MO blockers:** the DTLZ `cumprod`/`flip` v1-export issue is
  FIXED. Remaining: (1) NSGA3 `matrix_rank` export rejection at
  `src/evox_etl/algorithms/mo/nsga3.py:216` (12 cells: 4 per compiled
  backend — previously masked by the cumprod error); (2) iree-compile
  segfault (`Error code: -11`, stack in `libIREECompiler.so`) on NSGA2/MOEAD
  MO programs (15 cells: 8 iree-cuda + 7 iree-llvm-cpu); (3)
  MOEAD/DTLZ2/100x3x10 on iree-llvm-cpu dies at runtime (`ref is null`,
  `hal.buffer_view.create`).
- **CMA-ES `c_c` framework bug is FIXED** in both torch and etl (Hansen
  canonical formula); all 9 CMA-ES SO cells on all 6 backends were re-run.
- **CMAES/Ackley/100x10 divergence persists:** etl stalls on the Ackley
  plateau (best ≈ 20.5–20.6, rel_err ≈ 5.7e3 vs torch-cpu 0.0036) on all
  four etl backends; the c_c fix did not address it. torch-cuda Ackley
  1000x50/10000x100 also stall (RNG variance).
- **Parity (vs torch-cpu, 10% rel tolerance, near-zero rule):** SO 90 ok /
  66 not-ok / 24 skip (etl-numpy gens caps); MO 10 ok / 22 not-ok / 28 skip
  (backend errors). etl compiled backends are bit/last-ulp identical to
  each other; not-ok cells are the documented CMAES-Ackley divergence plus
  RNG-stream scatter on unconverged runs (see `BENCHMARK_RESULTS.md`).
- Details, tables and key numbers: see `BENCHMARK_RESULTS.md`.

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
