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
- **Numbers exist for:** SO on all six backends (torch-cpu, torch-cuda,
  etl-numpy, etl-iree-llvm-cpu, etl-iree-cuda, etl-xla-cuda — 36 cases each);
  MO on torch-cpu, torch-cuda, etl-numpy (12 cases each).
- **etl-numpy caps:** SO 1000x50 → 50 gens, 10000x100 → 10 gens, MO
  1000x3x30 → 20 gens (interpreter too slow / O(n²) whole-history PF
  ranking). Capped records keep metrics; parity is skipped.
- **etl-GPU MO is BLOCKED** (12/12 `BackendError` per compiled backend):
  stablehlo v1 exporter rejects `cumprod`/`flip` in
  `src/evox_etl/problems/numerical/dtlz.py`; no compile option exists.
- **CMA-ES `c_c > 2` framework bug** at pop ≥ 1000, dim ≥ 50 (Hansen
  parametrization → NaN covariance): 6 error records per etl backend,
  6 cusolver errors on torch-cuda, and 6 silently-NaN torch-cpu records.
- **CMAES/Ackley/100x10 divergence:** etl CMA-ES stalls on Ackley's
  zero-gradient plateau (≈20.4) vs torch ≈0.003; rel_err ≈ 6.5–6.6e3 on all
  four etl backends.
- **Parity (vs torch-cpu, 10% rel tolerance, near-zero rule)**: SO 84 ok /
  48 not-ok / 48 skip; MO 7 ok / 11 not-ok / 42 skip. etl compiled backends
  are bit/last-ulp identical to each other; not-ok cases are the documented
  CMAES-Ackley divergence plus RNG-stream scatter on unconverged runs.
- Details, tables and key numbers: see `BENCHMARK_RESULTS.md`.

## Notes for agents
- Result JSON schema and field semantics are documented in
  `results/README.md` — read it before touching the JSONs.
- `parity` blocks compare each non-torch-cpu record against the torch-cpu
  record with the same `case_id`; `ok` is `true`/`false`/`null` (null only
  for skips: backend error, baseline error, or gens cap). Near-zero pairs
  (both ≤ 1e-4) are ok by rule; the full semantics are stated in
  `BENCHMARK_RESULTS.md` §Parity verdicts. Do not hand-edit verdicts —
  recompute them with those rules instead.
- Never ship wrong numbers: every cell in `BENCHMARK_RESULTS.md` must trace
  back to a `results/*.json` record; error/NaN/capped records must be
  marked (`err`, `NaN (silent)`, `@Ng`) instead of shown as plain values.
- Known harness quirks (torch MO default-device workaround, per-generation
  PF filter instead of `EvalMonitor.get_pf_fitness`, xla `device=None`
  staging, MOEAD pop_size overwritten by Das-Dennis count, NSGA3
  non-reproducibility across GPU runs) are recorded in `results/README.md`
  and `BENCHMARK_RESULTS.md` — do not re-derive them.
