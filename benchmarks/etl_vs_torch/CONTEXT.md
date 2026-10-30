# benchmarks/etl_vs_torch — evoxc-backed EvoX benchmark harness + frozen 1.x comparison artefacts

## Intent
Live benchmark harness for the shipped functional `evox_etl` (`src/evox_etl/`, executed by the `evoxc` compiler): SO and MO suites, small → large scale, per-step wall-clock timings, Pareto-front metrics and a parity verdict against a frozen historical baseline.
The directory also preserves the 1.4-era torch-OOP-vs-etl comparison report, the code-style comparison and the `results/*.json` matrices as frozen historical experimental evidence.
No torch code exists in this tree: nothing here imports `torch`, `torch.*` or the deleted `evox` package.
The only implementation imports are `evox_etl` (from the repo `src/`) and `evoxc`, plus numpy/stdlib.

## API Surface
- `bench_common.py` — stdlib + numpy only, imported eagerly by `run.py` (so it must never import `evoxc`/`evox_etl`/torch at module scope). Holds the repo `src/` sys.path shim, case tables/seeds/scales, numpy `hv`/`igd` metrics, seeded DTLZ reference fronts, the backend id table + `etl_backend_spec`, the NVIDIA/ROCm env recipes with the one-shot re-exec, `init_with_retry` (evoxc compile retries), the JSON record schema, `save_results` + parity computation, and `finish_run` (PJRT-safe exit).
- `bench_so.py` — SO runner: PSO / DE / CMAES / OpenES × Sphere / Rastrigin / Ackley × 100x10, 1000x50, 10000x100; per-case error records instead of crashes.
- `bench_mo.py` — MO runner: NSGA2 / NSGA3 / MOEAD × DTLZ1 / DTLZ2 × 100x3x10, 1000x3x30; per-generation Pareto-front extraction (`_etl_pf_fitness`), HV/IGD computed by the numpy helpers.
- `run.py` — CLI dispatcher: `--backend <id> --suite {so,mo} [--gpu N] [--cases ...] [--gens-override N] [--out PATH] [--list]`; applies the GPU env recipe BEFORE the lazily imported runner touches any backend.
- `retag_parity.py` — stdlib-only maintenance tool: merges a partial `--out` run into `results/{suite}_{backend}.json` by `case_id` and recomputes parity verdicts (`--partial PATH [...] [--cap-note ...] [--set-note CASE=TEXT] [--note-override CASE=TEXT] --apply`); default is a dry run.
- `BENCHMARK_RESULTS.md` / `style_comparison.md` / `results/README.md` — frozen 1.x-era reports; each carries a provenance note at the top (numbers measured against the removed torch reference, `torch-*` ids gone, `etl-*` ids name the evoxc-backed implementations). Never re-measure into them.
- `results/` — frozen JSONs: A6000/CUDA matrix (`{so,mo}_{backend}.json` × 6 backends), MI300X/ROCm files (`{so,mo}_torch-rocm.json`, `{so,mo}_etl-xla-rocm.json`, `{so,mo}_etl-iree-rocm.json`, `{so,mo}_etl-iree-llvm-cpu_mi300x.json`) and `rocm_prefix_baseline.json`; schema and metric semantics in `results/README.md`.

## Backends
`bench_common.ALL_BACKENDS` is exactly these six ids (there is no torch backend and no `--scale` flag):
| id | vendor | evoxc `(backend, device)` |
|---|---|---|
| `etl-numpy` | — | `("numpy", None)` |
| `etl-iree-llvm-cpu` | — | `("iree", None)` |
| `etl-iree-cuda` | nvidia | `("iree", "cuda:0")` |
| `etl-iree-rocm` | rocm | `("iree", "rocm:0")` |
| `etl-xla-cuda` | nvidia | `("xla", "cuda:0")` |
| `etl-xla-rocm` | rocm | `("xla", "rocm:0")` |
The `etl-*` ids are FROZEN keys into the committed result JSONs — keep the names even though the implementation package is `evoxc` (nothing named `etl` is importable).
`CUDA_BACKENDS` / `ROCM_BACKENDS` drive the env recipe, the `--gpu` requirement and `backend_vendor`; `PJRT_BACKENDS` (the two xla ids) additionally select the `EVOXC_PJRT_PLUGIN` recipe and the `os._exit(0)` exit path.
Requesting a removed torch id fails fast: argparse rejects it with `invalid choice: 'torch-cuda' (choose from 'etl-numpy', ...)` and exit code 2 (the runners use `choices=ALL_BACKENDS`, `required=True`).
`torch-cpu` survives only as a LABEL: the parity baseline file `results/{suite}_torch-cpu.json` is read as JSON (never written, no torch involved).

## Constraints
- **No torch, no `evox`**: only `evox_etl` + `evoxc` + numpy + stdlib; `bench_common` must stay importable before the GPU env is set.
- **Backend-independent methodology**: case tables, seed 42, `DEFAULT_GENS=100`, `WARMUP_STEPS=2` and the per-step timed region are shared by all backends — do not weaken them per backend.
- **`hv` semantics**: the numpy `hv` mirrors `evox_etl.metrics.hv` — `points = np.maximum(ref - objs, 0.0)`, so points that do not dominate the reference point contribute nothing. Newly produced HV values for early/unconverged fronts are therefore LOWER than the committed `abs`-era numbers; never compare the two directly.
- **Never write smoke runs into `results/`**: the runners OVERWRITE `results/{suite}_{backend}.json` with only the cases they run, so partial runs use `--out /tmp/...` and are merged with `retag_parity.py`.
- **Frozen artefacts**: `results/**`, `BENCHMARK_RESULTS.md` and `style_comparison.md` keep their numbers; only provenance/commentary prose may be added.

## How to run
CPU-only (no extra toolchain): `/tmp/venv/bin/python benchmarks/etl_vs_torch/run.py --backend etl-numpy --suite so --cases PSO/Sphere/100x10 --gens-override 3 --out /tmp/smoke.json`.
List case ids with `--list`; `--suite mo` runs the MO suite the same way.
`--gpu N` is required for GPU ids and applies the vendor recipe through `ensure_backend_env` (re-execs once, marker `EVOX_BENCH_GPU_ENV`, keeps `sys.argv`):
- NVIDIA: `CUDA_VISIBLE_DEVICES=<id>`, cuDNN 9.8 dir prepended to `LD_LIBRARY_PATH`, `XLA_FLAGS=--xla_gpu_cuda_data_dir=...`, `EVOXC_PJRT_PLUGIN=<venv>/jax_plugins/xla_cuda12/xla_cuda_plugin.so`; `etl-xla-cuda` additionally LD_PRELOADs cuDNN.
- AMD ROCm: `HIP_VISIBLE_DEVICES`/`ROCR_VISIBLE_DEVICES`, and for the xla id `EVOXC_PJRT_PLUGIN=<venv>/jax_plugins/xla_rocm7/xla_rocm_plugin.so` plus `XLA_FLAGS=--xla_gpu_enable_command_buffer=` (HIP command buffers SIGSEGV in `PJRT_Executable_Execute`).
ROCm runs are serial, foreground and single-tenant; write them to the natural `*-rocm` file names, and a CPU-column refresh to `results/{suite}_etl-iree-llvm-cpu_mi300x.json` via `--out` so the A6000 file is not clobbered.

## Robustness
- **Compile retry**: `bench_common.init_with_retry` builds a fresh workflow and retries `init(seed)` up to `EVOXC_COMPILE_ATTEMPTS` (4) with a 1 s backoff, timings only the successful attempt as `compile_time_s`; required because iree MO compilation fails nondeterministically. `KeyboardInterrupt`/`SystemExit` are never retried and a hard SIGSEGV still kills the process.
- **PJRT teardown**: `save_results` flushes + fsyncs the JSON, then `finish_run` leaves via `os._exit(0)` for `PJRT_BACKENDS` so a completed xla run is never reported as a crash.

## Known Issues
- **iree MO compile blocker (current, no fix attempted):** `iree-compile` SIGSEGV on all MO programs (upstream while-loop compiler bug); `etl-xla-*` is the working compiled MO backend.
- **`etl-numpy` is an interpreter and is capped by convention, not code:** use `--gens-override 50` for SO 1000x50 and `10` for 10000x100 when producing numpy evidence, and mark the records `@Ng`.
- **Committed xla SO numbers are obsolete as GPU evidence:** they were produced by the old `("xla", None)` host-staging mapping; device-resident xla runs ~0.9-1.0 ms/step at 10000x100 (see the caveat note in `BENCHMARK_RESULTS.md`).
- **Timeouts/monitor history**: keep `full_sol_history`/`full_pop_history` off in benchmarks — enabling them copies `(pop, dim)` per step.

## Notes for Agents
- `results/README.md` is the schema/field-semantics reference — read it before touching any JSON.
- Per-case `parity` blocks compare against `results/{suite}_torch-cpu.json` by `case_id`; `ok` may be `true`/`false`/`null` (null only for skips: backend error, baseline error, gens cap), near-zero pairs are ok by rule, `PARITY_TOL = 0.10`. Recompute verdicts with `retag_parity.py` — never hand-edit them.
- Every number quoted in `BENCHMARK_RESULTS.md` must trace to a `results/*.json` record; error/capped records are marked `err`/`@Ng` rather than shown as plain values.
- GPU recipe paths reference the original measurement hosts (venv/plugin locations) — adjust locally rather than committing host-specific defaults.

## Routing Table
_(flat directory — no child subdirectories; `results/` holds only frozen JSON artefacts and its README)_
