# unit_test/problems/neuroevolution/ — Neuroevolution Problem Tests

## Intent
Integration tests for neuroevolution problems under `evox.problems.neuroevolution`. These tests exercise full neuroevolution pipelines (problem construction, forward/evaluate with LoRA-perturbed populations, fitness correctness) using lightweight inline MLP fixtures and deterministic `DataLoader`s.

## API Surface
- `test_virtual_lora_problem.py` — Tests for `VirtualLoRAProblem` (shape, sigma=0 center-loss correctness, determinism, seed-dependence, multi-layer forward, 1D/2D perturbation).

## Conventions
- Uses `unittest.TestCase`.
- Each `setUp` sets `torch.set_default_device(...)` (CPU fallback, no CUDA in dev env) and `torch.manual_seed(42)`.
- Inline helper fixtures defined in the test file: `make_simple_mlp()`, `make_data_loader()`, `count_params()`.
- Tests run with `PYTHONPATH=./src` to override the venv's stale editable install.

## Constraints
- No external dataset/model downloads in these tests — fixtures are synthetic tensors.
- Determinism tests account for the problem's internal `data_loader_iter` advancing between `evaluate` calls (use `n_batch_per_eval=-1` to iterate the full dataset each call).
