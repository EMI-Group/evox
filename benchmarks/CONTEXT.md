# benchmarks/ — performance benchmarking

## Intent
Holds the performance benchmarks of the EvoX 2.0 pre-release, all driving the `evoxc`-backed implementations.
The old PyTorch 1.x benchmark scripts (`pso.py`, `utils.py`, `test_base.py`, `benchmark_virtual_lora_es.py`, which used `torch.compile`/`vmap`/`cuda` memory probes) were removed with the torch package.
Nothing here is collected by pytest (`pyproject.toml` restricts testpaths to `unit_test/etl`).

## API Surface
- `etl_vs_torch/` — the cross-backend comparison harness (self-contained: `run.py` driver, `bench_common.py`, `bench_so.py`, `bench_mo.py`, committed result JSONs, `BENCHMARK_RESULTS.md`, `style_comparison.md`). See its own `CONTEXT.md`.
- This directory has no stand-alone scripts of its own any more.

## Constraints
- Benchmarks are ad-hoc/manual (`if __name__ == "__main__"`), not CI: no assertions or test framework here.
- Keep compiled-backend/GPU work inside `etl_vs_torch/` (that harness owns the backend matrix and its committed results).
- Do not reintroduce torch-only scripts: this repository's runtime dependencies are `evoxc` + `numpy` (compiled backends are optional extras).

## Routing Table
| Path | Description |
|---|---|
| `./etl_vs_torch` | Torch-vs-ETL benchmark harness: 9 backend ids × SO/MO suites × 3 scales, result JSONs + `BENCHMARK_RESULTS.md` + `style_comparison.md` |
