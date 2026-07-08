# benchmarks/

## Intent
Performance benchmarking suite for EvoX. Measures wall-clock time and PyTorch profiler traces for core evolutionary computation workflows, including eager vs `torch.compile` comparisons, vectorized (`vmap`) execution, and utility function throughput.

## API Surface
- **`test_base.py`** — Reusable benchmarking harness (`test()` function) for any EvoX `Algorithm`. Runs `StdWorkflow` + `EvalMonitor` on the `Sphere` problem. Tests both standard `compile` and `vmap`-based compiled steps. Optionally exports `torch._dynamo.explain` traces as markdown.
- **`pso.py`** — Standalone PSO benchmark. Compares three compilation modes (eager, default `compile`, `max-autotune-no-cudagraphs`) over 1000 steps on a 500-dim Sphere problem.
- **`utils.py`** — Benchmarks EvoX utility functions (`evox.utils.switch`). Tests basic compilation and large-scale `vmap` performance (1000×10000 tensors).
- **`benchmark_virtual_lora_es.py`** — Compares `VirtualES` (memory-efficient virtual population via on-demand perturbations, never materialized) against a naive `OpenES` baseline across a population sweep (16 → 16384). Uses a transformer-structured `nn.Sequential` model (`nn.Linear` + `nn.GELU`, ~29.7M params / 29,664,778 parameters) that is `VirtualProblem`-compatible. Also runs a SECOND flat-vector "virtual metric" experiment (`N_PARAMS=50_000_000`) that treats parameters as a flat vector and computes `mean(|center + sigma*noise|)` per individual with no model forward — isolating the virtual-population mechanism via pure memory-bandwidth. The shared `VectorMetricProblem` handles both the virtual tuple payload `(center, seeds, sigma)` and the naive materialized `(pop_size, n_params)` population; uses the fused `virtual_reduce_metric` op when available (try/except import), falling back to `_cpu_normal_noise`-generated noise. Measures per-step wall-clock time and peak memory (CUDA via `torch.cuda.max_memory_allocated` with `gc.collect()`+`empty_cache` cleanup before reset, CPU via `tracemalloc`). Records OOM configurations with `status: "oom"` and continues the sweep; saves results incrementally to `benchmarks/results/benchmark_results.json` (JSON: metadata + model-based `results` + `vector_metric_results`, each with per-pop-size naive/virtual entries). Generates matplotlib plots (time & memory) for BOTH experiments (`benchmark_{time,memory}.png` and `benchmark_metric_{time,memory}.png`), stopping each line at the last successful point before OOM (`_successful_prefix`); also exposes a standalone `plot_from_json()` to regenerate all plots from saved JSON. Auto-detects CUDA with CPU fallback.

## Constraints
- All benchmarks auto-detect CUDA and default to GPU when available, falling back to CPU.
- No external dependencies beyond PyTorch and EvoX itself.
- Designed for ad-hoc manual execution (`if __name__ == "__main__"`), not automated CI — no test frameworks or assertions.

## Routing Table
_(flat directory — no child subdirectories)_
