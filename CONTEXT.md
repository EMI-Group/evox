# EvoX — Distributed GPU-accelerated Evolutionary Computation Framework

## Intent
EvoX is a distributed GPU-accelerated evolutionary computation framework built on top of PyTorch. It provides a comprehensive suite of 50+ evolutionary algorithms and 100+ benchmark problems/environments, with a programming model designed for scalability across CPUs and GPUs.

**Core goals:**
- **High performance**: GPU acceleration via PyTorch tensorization, `torch.compile`, and `vmap` vectorization, achieving 100x+ speedups over CPU baselines.
- **All-in-one**: Single- and multi-objective optimization, meta-learning, hyperparameter optimization (HPO), and neuroevolution.
- **Easy-to-use**: PyTorch-native, hierarchical component model (`Algorithm` → `Problem` → `Workflow`), one-click install.

**Project metadata:** GPL-3.0-or-later, Python ≥3.10, PyTorch ≥2.6.0. Version 1.3.0 (current `pyproject.toml`).

## Repository Structure

```
evox/
├── src/evox/           ← Main Python package (the framework)
├── src/evox_ext/       ← Extension/plugin autoloading (PEP 420 namespace)
├── unit_test/          ← Unit test suite (mirrors src/evox/)
├── benchmarks/         ← Performance benchmark scripts
├── docs/               ← Sphinx documentation (ReadTheDocs, bilingual EN/ZH)
├── .github/            ← CI/CD workflows + PR template
├── pyproject.toml      ← Build config, deps, tool.ruff linting
├── MANIFEST.in         ← Source distribution manifest
├── flake.nix / flake.lock / dev_env_builder.nix  ← Nix dev environment
├── .pre-commit-config.yaml
├── .readthedocs.yaml
├── README.md / README_ZH.md
└── LICENSE
```

## Routing Table

| Area | Path | Description |
|---|---|---|
| **Main package** | `src/evox/` | Core framework: algorithms, problems, operators, workflows, metrics, utils, visualization |
| Core abstractions | `src/evox/core/` | `ModuleBase`, `Parameter`, `Mutable`, `compile`, `vmap`, `use_state` + component ABCs (`Algorithm`, `Problem`, `Workflow`, `Agent`, `Monitor`) |
| Algorithms | `src/evox/algorithms/` | 50+ EA implementations: SO (DE/ES/PSO families) + MO (NSGA2, NSGA3, RVEA, MOEAD, HypE, RVEAa) |
| Problems | `src/evox/problems/` | Benchmark problems: numerical (CEC2022, DTLZ, basic), neuroevolution (Brax, MuJoCo, supervised learning), HPO wrapper |
| Operators | `src/evox/operators/` | Genetic operators: selection, crossover, mutation, sampling (pure stateless tensor functions) |
| Workflows | `src/evox/workflows/` | `StdWorkflow` (standard optimization loop) + `EvalMonitor` (fitness/elite tracking) |
| Metrics | `src/evox/metrics/` | Multi-objective metrics: GD, HV, IGD |
| Triton kernels | `src/evox/triton_kernels/` | Optional Triton GPU kernels with PyTorch fallback; `register_triton_op` decorator, backend detection |
| Utilities | `src/evox/utils/` | JIT/vmap-compatible tensor ops, custom op registration, param↔vector conversion, PyTree re-exports |
| Visualization | `src/evox/vis_tools/` | Plotly-based interactive plots + EvoXVision (.exv) binary serialization |
| **Extensions** | `src/evox_ext/` | PEP 420 namespace-package plugin system; auto-discovers external algorithms, problems, operators, metrics, utils |
| **Tests** | `unit_test/` | `unittest`-based; mirrors `src/evox/` structure; tests eager, `torch.compile`, and `vmap` modes |
| **Benchmarks** | `benchmarks/` | PSO benchmark (eager vs compile vs max-autotune), `switch` micro-benchmark, reusable `test_base.py` |
| **Documentation** | `docs/` | Sphinx + shibuya theme; autodoc2 API docs; MyST Markdown tutorials; bilingual (gettext .po + manual ZH translations) |
| **CI/CD** | `.github/workflows/` | Python package build/test, PyPI publish, Ruff lint check, Discord bot notifications |

## Key Architectural Concepts

### Component Hierarchy
All framework components inherit from `ModuleBase` (extends `torch.nn.Module`):
```
ModuleBase
  ├── Algorithm   — Evolutionary search logic (step, init_step, evaluate proxy)
  ├── Problem     — Fitness evaluation (evaluate)
  ├── Workflow    — Orchestrates Algorithm + Problem + Monitor
  ├── Agent       — Individual agent for RL-style tasks
  └── Monitor     — Lifecycle callbacks for observation/logging
```

### Functional Programming Model
- `Parameter(value)` — marks hyperparameters (stateless, fixed during search)
- `Mutable(value)` — marks state tensors (population, fitness, velocities)
- `use_state` — converts module methods to functional form for `vmap`
- `compile` / `vmap` — PyTorch wrappers with scalar-index graph-break workarounds

### Extension System
External packages install into the `evox_ext` namespace package. At `import evox` time, `auto_load_extensions()` discovers and merges them into the corresponding `evox.*` modules (algorithms, problems, operators, metrics, utils).

### Triton Kernel Integration
Triton provides hand-written GPU kernels for performance-critical operations. The infrastructure is **optional** — Triton is an optional dependency (`pip install evox[triton]`). When available, `register_triton_op` registers both a PyTorch fallback (runs everywhere) and a Triton CUDA kernel. PyTorch's dispatcher auto-routes CUDA → Triton, CPU/other → PyTorch. Without Triton installed, only the PyTorch fallback is used and everything works normally.

## Constraints
- **Pure PyTorch**: No NumPy in framework code. All tensors must be PyTorch tensors for GPU compatibility.
- **Compile-friendly**: Operators and algorithms must be `torch.compile`/`vmap` compatible. No Python control flow depending on tensor values.
- **Minimization semantics**: All algorithms minimize internally; `StdWorkflow` applies `opt_direction` transforms for maximization.
- **Monitor outside jit**: Monitors run outside the compiled graph. Use token-passing patterns for compile-safe history.
- **Bilingual docs**: Documentation supports English and Simplified Chinese via ReadTheDocs language switching.
