# Operators (`src/evox/operators/`)

## Intent
The **operators** module provides the core genetic/evolutionary building blocks for EvoX — a GPU-accelerated evolutionary computation framework on PyTorch. This module contains four families of operators that algorithms compose to form complete evolutionary pipelines: **selection**, **crossover**, **mutation**, and **sampling**.

All operators share a common design philosophy:
- **Pure functions** operating on PyTorch tensors — stateless, no `ModuleBase` inheritance, no retained randomness.
- **Device-aware** — random tensors are created on the input tensor's device for seamless GPU execution.
- **Fully vectorized** — no Python loops over individuals; all operations are batched tensor operations.
- **`torch.compile` and `vmap` friendly** — designed to compose cleanly with `evox.core` transformation primitives.

Operators are consumed by algorithms (`src/evox/algorithms/`) which are `ModuleBase` subclasses that orchestrate these pure functions into iterative evolutionary loops.

## API Surface

The top-level `evox.operators` namespace exports:
- **Submodules**: `crossover`, `mutation`, `sampling`, `selection`
- **Re-exports from selection**: `crowding_distance`, `non_dominate_rank`

```python
from evox.operators import crossover, mutation, sampling, selection
from evox.operators import crowding_distance, non_dominate_rank
```

## Routing Table

| Directory | Purpose |
|---|---|
| `selection/` | Selection operators — choosing which individuals survive/reproduce: non-dominated sorting (Pareto ranking), crowding distance, tournament selection, RVEA reference-vector-guided selection, and personal-best selection |
| `crossover/` | Crossover (recombination) operators — combining parent solutions to produce offspring: Simulated Binary Crossover (SBX) for real-valued GAs, and Differential Evolution (DE) building blocks (differential sum, binary/exponential crossover, arithmetic recombination) |
| `mutation/` | Mutation operators — perturbing individuals to maintain diversity: Polynomial Mutation (PM), the canonical real-valued mutation operator |
| `sampling/` | Sampling/initialization operators — generating initial populations and reference vectors: uniform simplex lattice (Das & Dennis), grid sampling, Latin Hypercube Sampling |

## Relationship to `evox.core`

- Operators are **stateless building blocks** — they are plain functions, not `ModuleBase` subclasses.
- Algorithms (`evox.algorithms/`) compose operators within `ModuleBase` subclasses that manage state, iteration, and the full evolutionary loop.
- Core transformations (`evox.core.compile`, `evox.core.vmap`, `evox.core.use_state`) can wrap operator calls for JIT compilation and vectorized batching.
- Utility functions from `evox.utils` (e.g., `lexsort`, `register_vmap_op`, `maximum`, `minimum`, `clamp_float`) are used throughout the operators for efficient tensor operations.

## Constraints

- Operators must remain **pure and stateless** — no `self`, no internal buffers, no random seeds carried across calls.
- All tensor inputs must reside on the same device; outputs inherit the input device.
- Operators assume **minimization** semantics (lower fitness = better).
- New operators should follow the existing pattern: a single public function per file, re-exported via `__init__.py` and `__all__`.
