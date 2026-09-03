# benchmarks/etl_vs_torch — torch evox vs evox_etl comparison

## Intent
Benchmarks comparing the torch OOP evox (`src/evox/`) against the functional
`evox_etl` (`src/evox_etl/`) on CPU and GPU, small → large scale, plus a code
style/cleanliness comparison report. See `../../src/evox_etl/DESIGN.md` §7.

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
