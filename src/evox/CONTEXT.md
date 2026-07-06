# `src/evox/` — Main Package Entry Point

## Intent
This is the top-level Python package for the EvoX framework. `import evox` gives users the full public API: core abstractions (`ModuleBase`, `Parameter`, `Mutable`, `compile`, `use_state`, `vmap`) and submodule access to algorithms, problems, operators, workflows, metrics, utilities, and visualization tools. It also triggers the extension autoloading mechanism.

## API Surface

### Re-exported Core Symbols (accessible as `evox.<name>`)
| Symbol | Source | Purpose |
|---|---|---|
| `ModuleBase` | `evox.core` | Base module for all framework components |
| `Parameter` | `evox.core` | Hyperparameter wrapper (`nn.Parameter`, requires_grad=False) |
| `Mutable` | `evox.core` | Mutable state tensor wrapper (`nn.Buffer`) |
| `compile` | `evox.core` | Fixed `torch.compile` wrapper |
| `use_state` | `evox.core` | Functional transform for stateful modules |
| `vmap` | `evox.core` | Fixed `torch.vmap` wrapper |

### Submodule Access (accessible as `evox.<module>`)
| Module | What it contains |
|---|---|
| `evox.algorithms` | 50+ EA implementations (SO: DE/ES/PSO families; MO: NSGA2, NSGA3, RVEA, MOEAD, HypE, RVEAa) |
| `evox.problems` | Benchmark problems: numerical, neuroevolution, HPO wrapper |
| `evox.operators` | Genetic operators: selection, crossover, mutation, sampling |
| `evox.workflows` | `StdWorkflow` + `EvalMonitor` |
| `evox.metrics` | MO metrics: GD, HV, IGD |
| `evox.utils` | JIT/vmap-compatible tensor ops, op registration, param↔vector conversion |
| `evox.vis_tools` | Plotly-based visualization + EvoXVision binary format |
| `evox.core` | Foundation types: `ModuleBase`, components, `compile`, `vmap`, `use_state` |

### Extension Loading
On import, `auto_load_extensions()` is called, which discovers packages in the `evox_ext` namespace and merges them into the corresponding `evox.*` modules. This allows third-party packages to register new algorithms, problems, operators, metrics, and utilities without modifying EvoX source.

## Constraints
- `import evox` must succeed with only `torch` and `numpy` installed (core dependencies). Optional dependencies (plotly, pandas, brax, etc.) are only needed for specific submodules.
- `__all__` defines the public API; everything else is internal.
- Extensions are loaded silently — missing `evox_ext.*` packages do not cause import errors.

## Routing Table
| Area | Path | Description |
|---|---|---|
| Core abstractions | `core/` | `ModuleBase`, `Parameter`, `Mutable`, `compile`, `vmap`, `use_state`, component ABCs |
| Algorithms | `algorithms/` | SO (DE/ES/PSO families) and MO evolutionary algorithms |
| Problems | `problems/` | Numerical, neuroevolution, and HPO problem definitions |
| Operators | `operators/` | Pure stateless genetic operators |
| Workflows | `workflows/` | Optimization orchestration and monitoring |
| Metrics | `metrics/` | Multi-objective quality indicators |
| Utilities | `utils/` | JIT/vmap-compatible ops and helpers |
| Visualization | `vis_tools/` | Plotly plots and EvoXVision serialization |
