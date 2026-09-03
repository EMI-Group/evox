# benchmarks/etl_vs_torch — torch evox vs evox_etl comparison

## Intent
Benchmarks comparing the torch OOP evox (`src/evox/`) against the functional
`evox_etl` (`src/evox_etl/`) on CPU and GPU, small → large scale, plus a code
style/cleanliness comparison report. See `../../src/evox_etl/DESIGN.md` §7.

## Status: IMPLEMENTED (all six backends smoke-validated)
- `bench_common.py` — sys.path shim, case tables, numpy metric helpers (hv/igd
  + seeded DTLZ ref fronts), JSON schema, parity vs torch-cpu, GPU env recipe.
- `bench_so.py` / `bench_mo.py` — per-backend runners (torch eager, etl
  numpy/iree/xla), resilient per-case error records, compact per-case lines.
- `run.py` — CLI dispatcher: `--backend --suite {so,mo} [--gpu N] [--cases ...]
  [--gens-override N] [--list]`; sets GPU env BEFORE backend imports.
- `results/README.md` — schema + naming convention (run artifacts are NOT
  committed; smoke runs use `--out <tmp path>`).

## Discovered quirks (recorded in the code + results/README.md)
- etl StdWorkflow cannot auto-complete EvalMonitorConfig for CMA-ES/OpenES
  (their states have `y`/`noise`, not `population`/`pop`) — SO runs pass
  pop_size/dim explicitly; MO relies on auto-completion (works for
  NSGA2/NSGA3/MOEAD).
- torch algorithm constructors need explicit `device=` (their plain-attribute
  lb/ub stay on the default device otherwise → cuda step crashes).
- The venv etl xla adapter rejects non-cpu devices at load ("supports only
  CPU devices"); the CUDA plugin executes on the visible GPU anyway →
  etl-xla-cuda runs with device=None.
- The xla PJRT plugin (compiled vs cuDNN 9.8) has DT_RPATH → the venv's pip
  cuDNN 9.1 wins over LD_LIBRARY_PATH; the harness prepends
  /mnt/local-ssd/bchuang/cudnn-xla/lib/libcudnn.so.9 to LD_PRELOAD and
  re-execs the process (loader reads LD_PRELOAD at process start only).
- MOEAD overwrites the requested pop_size with the Das-Dennis count on both
  sides (recorded in the JSON `note`).

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
