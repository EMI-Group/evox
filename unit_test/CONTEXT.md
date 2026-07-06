# unit_test/ — EvoX Unit Test Suite

## Intent
Mirrors the `src/evox/` package structure to provide comprehensive unit tests for the EvoX distributed GPU-accelerated evolutionary computation framework. Tests verify algorithm correctness, operator behavior, problem evaluations, utility functions, workflow execution, and core engine features (JIT compilation, vmap vectorization, state dict serialization).

## Structure & Mapping to Source

| Test Directory | Source Module (`evox.*`) | Purpose |
|---|---|---|
| `algorithms/` | `evox.algorithms` | Algorithm correctness across 3 execution modes |
| `core/` | `evox.core` | Core engine: JIT utilities, index fix for vmap |
| `operators/` | `evox.operators` | Selection, crossover, mutation operators |
| `problems/` | `evox.problems` | Numerical benchmarks, neuroevolution, HPO wrappers |
| `utils/` | `evox.utils` | Utility functions (switch, ParamsAndVector) |
| `workflows/` | `evox.workflows` | StdWorkflow including distributed execution |

## Testing Conventions

### Framework & Device
- Uses Python `unittest` (`TestCase` / `unittest.TestCase`).
- GPU-first: `torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")` in every `setUp`.
- `torch.manual_seed(42)` is standard for reproducibility.

### Three Execution Modes (Algorithm Tests)
Algorithm tests verify behavior across three modes, defined in `algorithms/test_base.py`:
1. **Standard** (`run_algorithm`) — Plain eager execution with state-dict save/restore.
2. **Compiled** (`run_compiled_algorithm`) — `torch.compile(workflow.step)` for JIT.
3. **Vectorized** (`run_vmap_algorithm`) — `vmap` + `torch.func.stack_module_state` for batched parallel execution.

The `run_all` convenience method runs all three. Multi-objective algorithm tests (`test_moea.py`) define a parallel `MOTestBase` with the same three modes plus Pareto-front verification.

### Test File Organization
- `test_*.py` naming convention. Empty `__init__.py` in every subdirectory.
- Algorithm tests follow a `setUp` → per-variant `test_*` method pattern, each creating the algorithm and calling `run_all`.
- Problem tests instantiate multiple problem variants in a loop and validate shape, values, or compare against reference implementations.
- Operator tests verify correctness, then test under `torch.compile` and `vmap`.
- Workflow tests use lightweight inline `BasicAlgorithm` / `BasicProblem` definitions to minimize external dependencies.

### Inline Fixtures
No shared `conftest.py` or `fixtures/` directory. Test dependencies are defined inline:
- `algorithms/test_base.py` — `Sphere` problem, `TestBase` class with all three execution helpers.
- `workflows/test_std_workflow.py` — `BasicAlgorithm`, `BasicProblem`, `BasicMOPProblem`, `BasicMOPAlgorithm`.
- `problems/test_hpo_wrapper.py` — `BasicAlgorithm`, `BasicProblem` with `Parameter` for HPO testing.
- `problems/CEC2022_by_P_N_Suganthan.py` — External reference implementation (from P-N-Suganthan/2022-SO-BO) used only to validate EvoX's CEC2022 implementation.

### Special Test Files
- `problems/CEC2022_by_P_N_Suganthan.py` — **Not a test file.** Reference implementation for cross-validation of `CEC2022` problems. Uses NumPy (not PyTorch), called by `test_cec2022.py`.
- `problems/test_brax.py` / `test_mujoco.py` / `test_supervised_learning.py` — Integration-heavy tests that exercise full neuroevolution pipelines (algorithm + problem + workflow + monitor). These download datasets/models and can be slow.

## Constraints
- Tests should run on both CPU and CUDA devices.
- Algorithm tests should verify state-dict round-trip (save → run → restore).
- New algorithm variants must be tested in all three execution modes (standard, compiled, vmap).
- Problem correctness tests should validate against known reference values when available.
