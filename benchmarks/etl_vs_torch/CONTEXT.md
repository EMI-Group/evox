# benchmarks/etl_vs_torch — torch evox vs evox_etl comparison

## Intent
Benchmarks comparing the torch OOP evox (`src/evox/`) against the functional
`evox_etl` (`src/evox_etl/`) on CPU and GPU, small → large scale, plus a code
style/cleanliness comparison report. See `../../src/evox_etl/DESIGN.md` §7.

## Status: IMPLEMENTED (all six backends smoke-validated; torch-cuda full matrix committed)
- `bench_common.py` — sys.path shim, case tables, numpy metric helpers (hv/igd
  + seeded DTLZ ref fronts), JSON schema, parity vs torch-cpu, GPU env recipe.
- `bench_so.py` / `bench_mo.py` — per-backend runners (torch eager, etl
  numpy/iree/xla), resilient per-case error records, compact per-case lines.
- `run.py` — CLI dispatcher: `--backend --suite {so,mo} [--gpu N] [--cases ...]
  [--gens-override N] [--list]`; sets GPU env BEFORE backend imports.
- `results/README.md` — schema + naming convention (run artifacts are NOT
  committed; smoke runs use `--out <tmp path>`).
- Full torch-cuda matrix (GPU 0, 100 gens, seed 42) committed:
  `results/so_torch-cuda.json` (36 cases), `results/mo_torch-cuda.json` (12).

## Discovered quirks (recorded in the code + results/README.md)
- etl StdWorkflow cannot auto-complete EvalMonitorConfig for CMA-ES/OpenES
  (their states have `y`/`noise`, not `population`/`pop`) — SO runs pass
  pop_size/dim explicitly; MO relies on auto-completion (works for
  NSGA2/NSGA3/MOEAD).
- torch algorithm constructors need explicit `device=` (their plain-attribute
  lb/ub stay on the default device otherwise → cuda step crashes).
- The torch MO algorithms allocate tensors without an explicit device:
  NSGA2/NSGA3/MOEAD population init uses the raw (cpu) lb/ub constructor args,
  NSGA3's `vmap_get_table_row` indexes with `torch.arange` on
  `torch.get_default_device()`, and `uniform_sampling` builds reference
  points/weights on the default device. The MO torch path therefore runs each
  case with `torch.set_default_device(device)` set and lb/ub created on the
  target device (`bench_mo.run_torch_case`); torch-cpu is unaffected.
- `EvalMonitor.get_pf_fitness` materializes O(n²) domination matrices over the
  whole history (`non_dominate_rank`) — at 1000-pop × 100 gens (n≈100k) that
  needs ~280 GB and the process gets OOM-killed (cpu and cuda alike).
  `bench_mo._torch_pf_fitness` replaces it with per-generation fronts +
  an incremental numpy Pareto filter (mathematically identical, verified).
- **evox CMA-ES framework bug** (out of harness scope, reported upstream):
  `c_c = (mu_eff+2)/(dim+4+2*mu_eff/dim)` exceeds 2 at large mu_eff/dim
  (pop≥1000, dim≥50) → `torch.sqrt(c_c*(2-c_c)*mu_eff)` is NaN → p_c NaN →
  covariance all-NaN. On cuda, cusolver eigh raises `_LinAlgError` (error
  code 51/101) → the 6 large-scale CMA-ES torch-cuda cases are error records
  with a `note`; on CPU, LAPACK silently propagates the NaN (silently wrong
  results). Secondary bug at the same site: `p_c @ p_c.T` on the 1-D p_c is
  an inner product (scalar broadcast over C), not the intended rank-1 update.
- NSGA3 runs are not bit-reproducible across identical-seed GPU runs
  (non-deterministic reduction/sort kernels): hv/igd/n_pf vary ~1% run to run.
- The venv etl xla adapter rejects non-cpu devices at load ("supports only
  CPU devices"); the CUDA plugin executes on the visible GPU anyway →
  etl-xla-cuda runs with device=None.
- The xla PJRT plugin (compiled vs cuDNN 9.8) has DT_RPATH → the venv's pip
  cuDNN 9.1 wins over LD_LIBRARY_PATH; the harness prepends
  /mnt/local-ssd/bchuang/cudnn-xla/lib/libcudnn.so.9 to LD_PRELOAD and
  re-execs the process (loader reads LD_PRELOAD at process start only).
- MOEAD overwrites the requested pop_size with the Das-Dennis count on both
  sides (recorded in the JSON `note`).
- MOEAD torch is python-loop bound: ~3.5 s/step at 1000x3x30 (~6 min per
  case), vs ~0.27 s/step at 100x3x10.

## Plan (binding)
- Runners per backend: `torch-cpu`, `torch-cuda`, `etl-numpy` (CPU interpreter),
  `etl-iree-llvm-cpu`, `etl-iree-cuda`, `etl-xla-cuda` (xla once the etl adapter
  fixes land — otherwise report iree-cuda only).
- SO: PSO, DE, CMA-ES, OpenES on Sphere/Rastrigin/Ackley;
  scales (pop, dim): (100,10) small, (1000,50) medium, (10000,100) large; 100 gens.
- MO: NSGA2, NSGA3, MOEAD on DTLZ1/DTLZ2; scales (pop, obj): (100,3), (1000,3);
  dim 10/30; 100 gens.
- Same seeds and configurations across backends; verify final fitness parity within
  tolerance; measure per-step wall time + total; output markdown tables.
- Also produce `style_comparison.md`: code style/cleanliness pros & cons
  (torch OOP vs etl functional), based on the ported code.
- Venv: `/mnt/local-ssd/bchuang/evox/.venv/bin/python`. GPUs: scan `nvidia-smi`.
- IREE cuda constraints (from etl CONTEXT): default opt level; device
  `etl.core.Device("cuda", N)`; cuda inputs must be explicitly `.to(device)`.
