# benchmarks/

## Intent
Performance benchmarking suite for EvoX. Measures wall-clock time and PyTorch profiler traces for core evolutionary computation workflows, including eager vs `torch.compile` comparisons, vectorized (`vmap`) execution, and utility function throughput.

## API Surface
- **`test_base.py`** — Reusable benchmarking harness (`test()` function) for any EvoX `Algorithm`. Runs `StdWorkflow` + `EvalMonitor` on the `Sphere` problem. Tests both standard `compile` and `vmap`-based compiled steps. Optionally exports `torch._dynamo.explain` traces as markdown.
- **`pso.py`** — Standalone PSO benchmark. Compares three compilation modes (eager, default `compile`, `max-autotune-no-cudagraphs`) over 1000 steps on a 500-dim Sphere problem.
- **`utils.py`** — Benchmarks EvoX utility functions (`evox.utils.switch`). Tests basic compilation and large-scale `vmap` performance (1000×10000 tensors).
- **`benchmark_virtual_lora_es.py`** — Compares `VirtualLoRAES` (memory-efficient virtual population via on-demand LoRA perturbations) against a naive `OpenES` baseline across a population sweep (16 → 131 072). Uses a transformer-structured `nn.Sequential` model (`nn.Linear` + `nn.GELU`, ~940K params) that is `VirtualLoRAProblem`-compatible. Measures per-step wall-clock time and peak memory (CUDA via `torch.cuda.max_memory_allocated`, CPU via `tracemalloc`). Records OOM configurations with `status: "oom"` and continues the sweep; saves results incrementally to `benchmarks/results/benchmark_results.json` (JSON: metadata + per-pop-size naive/virtual entries). Generates two matplotlib plots (`benchmark_time.png`, `benchmark_memory.png`) of time/memory vs population size (log2/log), stopping the line at the last successful point before OOM; also exposes a standalone `plot_from_json()` to regenerate plots from saved JSON. Auto-detects CUDA with CPU fallback.

## Constraints
- All benchmarks auto-detect CUDA and default to GPU when available, falling back to CPU.
- No external dependencies beyond PyTorch and EvoX itself.
- Designed for ad-hoc manual execution (`if __name__ == "__main__"`), not automated CI — no test frameworks or assertions.

## Routing Table
_(flat directory — no child subdirectories)_
