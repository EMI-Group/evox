# benchmarks/

## Intent
Performance benchmarking for EvoX 2.0, measuring the shipped functional `evox_etl` implementation (`src/evox_etl/`, executed by the `evoxc` compiler) rather than the removed PyTorch 1.x OOP implementation.
Everything here measures the real optimization loop: per-step wall-clock time, compile time, Pareto-front metrics and JSON result records.
Benchmarks are ad-hoc manual runs (`if __name__ == "__main__"` / small CLIs), never CI assertions.

## API Surface
- `etl_vs_torch/` — the benchmark tree: an evoxc-backed harness (`bench_common.py`, `bench_so.py`, `bench_mo.py`, `run.py`, `retag_parity.py`) plus the frozen 1.4-era comparison artefacts (`BENCHMARK_RESULTS.md`, `style_comparison.md`, `results/**`). See its `CONTEXT.md`. The directory name and the `etl-*` backend ids are frozen historical keys, not the implementation package name.

## Constraints
- **No torch**: nothing under `benchmarks/` imports `torch`, `torch.*` or the deleted `evox` package — the harness imports `evox_etl` + `evoxc` + numpy + stdlib only.
- **Explicit backends**: no auto device detection — the backend id decides the `evoxc` backend/device, and GPU runs take `--gpu N` (the env recipe is applied before any backend import).
- **No PyTorch/NumPy-only shortcuts in methodology**: keep per-step timing, fixed seeds, warmup steps and the small → large scale sweep identical across backends.
- **Frozen artefacts stay frozen**: committed result JSONs and reports under `etl_vs_torch/` keep their historical numbers; only provenance prose may be added.

## Routing Table
| Area | Path | Description |
|---|---|---|
| evoxc-vs-baseline benchmark harness | `./benchmarks/etl_vs_torch/` | Live SO/MO harness for `evox_etl` on the `evoxc` backends (numpy / iree / xla), plus the frozen 1.x torch-vs-etl comparison report, style comparison and result JSON matrices |
