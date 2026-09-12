# `src/evox/algorithms/` — Evolutionary Algorithm Implementations

## Intent
This directory contains all evolutionary algorithm implementations for the EvoX framework. Every algorithm is a concrete subclass of `evox.core.Algorithm` (which extends `torch.nn.Module` via `ModuleBase`). Algorithms are fully tensorized and GPU-accelerated via PyTorch.

## API Surface — The `Algorithm` Contract
All algorithms follow a common lifecycle defined by `evox.core.Algorithm`:

| Method | Purpose |
|---|---|
| `__init__(...)` | Configure hyperparameters (`Parameter`), initialize state tensors (`Mutable`), set up population |
| `init_step()` | First-step evaluation; typically calls `self.evaluate(pop)` and does initial bookkeeping |
| `step()` | Core iteration: generate offspring, evaluate, select survivors |
| `record_step()` | Optional — returns a dict of tensors for monitoring (default: `{"pop": self.pop, "fit": self.fit}`) |

**Key design patterns:**
- **`Mutable`** wraps tensors that carry algorithmic *state* (population, fitness, velocities, etc.). These are traced by the workflow for vectorization (`vmap`) and compilation (`torch.compile`).
- **`Parameter`** wraps hyperparameters (learning rates, weights, probabilities). Also traced.
- **`self.evaluate(pop)`** is a proxy injected by the workflow — not defined by the algorithm. It delegates to `Problem.evaluate`.
- Most algorithms use genetic operators from `evox.operators` (crossover, mutation, selection, sampling).

## Directory Split: SO vs. MO

```
algorithms/
├── __init__.py          # Re-exports all SO + MO algorithms in a flat namespace
├── so/                  # Single-objective optimization algorithms
│   ├── de_variants/     # Differential Evolution family
│   ├── es_variants/     # Evolution Strategies family
│   └── pso_variants/    # Particle Swarm Optimization family
└── mo/                  # Multi-objective optimization algorithms
    ├── hype.py, moead.py, nsga2.py, nsga3.py, rvea.py, rveaa.py
```

## Routing Table

| Area | Path | Description |
|---|---|---|
| SO — DE variants | `so/de_variants/` | DE, SHADE, CoDE, SaDE, ODE, JaDE |
| SO — ES variants | `so/es_variants/` | OpenES, XNES, SeparableNES, DES, SNES, ARS, ASEBO, PersistentES, NoiseReuseES, GuidedES, ESMC, CMAES, VirtualES, VirtualLoRAES (alias — see note) |
| SO — PSO variants | `so/pso_variants/` | CLPSO, CSO, DMSPSOEL, FSPSO, PSO, SLPSOGS, SLPSOUS |
| MO — all algorithms | `mo/` | RVEA, RVEAa, MOEAD, NSGA2, NSGA3, HypE |

## Algorithm Families at a Glance

### Single-Objective (`so/`)

**DE Variants** (`so/de_variants/`) — Population-based stochastic optimizers using differential mutation and crossover. All operate on bounded continuous spaces (`lb`/`ub`).
- `DE` — Classic differential evolution (rand/best base vector, configurable difference vectors, binomial crossover)
- `SHADE` — Success-history based adaptive DE (auto-tunes F and CR from a historical memory)
- `CoDE` — Composite DE (combines multiple mutation strategies and parameter settings)
- `SaDE` — Self-adaptive DE (learns strategy probabilities online)
- `ODE` — Opposition-based DE (uses opposite points to improve exploration)
- `JaDE` — Adaptive DE with optional external archive (similar to SHADE with archive)

**ES Variants** (`so/es_variants/`) — Evolution Strategies that sample around a distribution center and follow the natural gradient. Most operate on unbounded spaces (center-init only).
- `OpenES` — Simple ES with Adam optimizer option, mirrored sampling
- `CMAES` — Covariance Matrix Adaptation ES (full covariance matrix adaptation)
- `XNES` / `SeparableNES` — Exponential Natural Evolution Strategies (full and separable covariance)
- `SNES` — Separable NES (diagonal covariance only)
- `DES` — Diagonal Evolution Strategy
- `ARS` — Augmented Random Search
- `ASEBO` — Adaptive ES with Bayesian Optimization
- `PersistentES` — ES with persistent noise perturbations across generations
- `NoiseReuseES` — Reuses noise samples for efficiency
- `GuidedES` — Guided ES with surrogate gradient
- `ESMC` — ES with Monte Carlo gradient estimation
- Shared utilities: `sort_utils.py` (fitness-based sorting), `adam_step.py` (Adam update for ES centers)

**PSO Variants** (`so/pso_variants/`) — Swarm intelligence using velocity-position updates with personal/global bests.
- `PSO` — Classic Particle Swarm Optimization (inertia + cognitive + social components)
- `CLPSO` — Comprehensive Learning PSO (learns from all particles' personal bests)
- `CSO` — Competitive Swarm Optimizer (pairwise competition rather than global best)
- `DMSPSOEL` — Dynamic Multi-Swarm PSO with Ensemble Learning
- `FSPSO` — Fitness-based PSO
- `SLPSOGS` / `SLPSOUS` — Social Learning PSO (global and uniform strategies)
- Shared utility: `utils.py` (e.g., `min_by` for finding global best)

### Multi-Objective (`mo/`)
All MO algorithms produce Pareto-front approximations with multiple objectives. They compose operators from `evox.operators` (crossover, mutation, selection, sampling) and track multi-dimensional fitness (`n_objs`). Common pattern: generate offspring via selection→crossover→mutation, evaluate, then merge and environmental-select.
- `NSGA2` — Non-dominated Sorting GA II (non-dominated ranking + crowding distance)
- `NSGA3` — NSGA-III (reference-point-based selection for many-objective problems)
- `RVEA` — Reference Vector Guided EA (angle-penalized distance via reference vectors)
- `RVEAa` — Adaptive RVEA (auto-adjusts reference vectors during search)
- `MOEAD` — MOEA based on Decomposition (scalarizes objectives via weight vectors)
- `HypE` — Hypervolume Estimation algorithm (uses Monte Carlo hypervolume approximation)

## Algorithm Counts (current)
Derived from the `__all__` export lists and `rg -n "^class " src/evox/algorithms` (both currently yield 33):
- **Single-objective (`so/`)**: 27 — DE variants 6, ES variants 14, PSO variants 7.
- **Multi-objective (`mo/`)**: 6.
- The ES variant `VirtualLoRAES` is exported as an **alias of `VirtualES`** (`virtual_es.py`: `VirtualLoRAES = VirtualES`), so the number of *distinct* classes reachable via `evox.algorithms.<Name>` is **32** (SO 26, MO 6).
- A genuine, distinct `VirtualLoRAES` (LoRA) implementation exists in `so/es_variants/virtual_lora_es.py` but is shadowed by the alias and is reachable only via its direct module path — so 33 distinct algorithm classes physically exist but only 32 are distinct through the package API.
- No abstract base or helper/mixin classes are defined under `algorithms/`; the only non-algorithm symbols are standalone helper functions (e.g. `min_by`, `sort_by_key`, `adam_single_tensor`, `pbi`, `cal_hv`).
- README.md, README_ZH.md, and the docs site state "30+ algorithms" built into EvoX and "140+ algorithms" across the EvoX ecosystem (EvoX plus ~13 sibling EMI-Group GPU-accelerated EC libraries).
- This package ships 33 algorithm classes, 32 distinct through the package API (see the points above).
- External `evox_ext` packages can add more, but they are not part of this repository.

## Constraints
- All algorithms are **pure PyTorch** — no NumPy, no CPU-bound loops.
- State variables must use `Mutable`; hyperparameters must use `Parameter`.
- `step()` must be stateless in the `torch.compile` sense — no Python control flow depending on tensor values.
- The `evaluate()` method is **not** defined here; it is set externally by the workflow as a proxy to `Problem.evaluate`.
- New algorithms should subclass `evox.core.Algorithm` and follow the `init_step`/`step` lifecycle.
