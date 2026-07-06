# `src/evox/algorithms/mo/` — Multi-Objective Evolutionary Algorithms

## Intent
This directory contains tensorized multi-objective evolutionary algorithm (MOEA) implementations for the EvoX framework. All algorithms produce Pareto-front approximations and handle multi-dimensional fitness (`n_objs`). Each is a concrete subclass of `evox.core.Algorithm` and is fully GPU-accelerated via PyTorch.

## API Surface

Six MOEAs, each in a standalone file, exported flat via `__init__.py`:

| File | Class | Family |
|---|---|---|
| `nsga2.py` | `NSGA2` | Non-dominated Sorting GA II |
| `nsga3.py` | `NSGA3` | NSGA-III (reference-point-based, many-objective) |
| `rvea.py` | `RVEA` | Reference Vector Guided EA |
| `rveaa.py` | `RVEAa` | Adaptive RVEA (with reference vector regeneration) |
| `moead.py` | `MOEAD` | MOEA based on Decomposition |
| `hype.py` | `HypE` | Hypervolume Estimation (Monte Carlo) |

All classes follow the `Algorithm` lifecycle: `__init__` → `init_step` → `step`. State tensors use `Mutable`; hyperparameters use `Parameter`. The `self.evaluate()` proxy is injected by the workflow at runtime.

## Shared Operator Dependencies

All MO algorithms import genetic operators from `evox.operators`:

| Operator | Source | Used By |
|---|---|---|
| `simulated_binary` | `evox.operators.crossover` | NSGA2, NSGA3, RVEA, RVEAa, HypE |
| `simulated_binary_half` | `evox.operators.crossover` | MOEAD (only) |
| `polynomial_mutation` | `evox.operators.mutation` | ALL |
| `tournament_selection_multifit` | `evox.operators.selection` | NSGA2, NSGA3 |
| `tournament_selection` | `evox.operators.selection` | HypE (single-fitness variant) |
| `non_dominate_rank` | `evox.operators.selection` | NSGA3, HypE, RVEAa |
| `nd_environmental_selection` | `evox.operators.selection` | NSGA2 |
| `ref_vec_guided` | `evox.operators.selection` | RVEA, RVEAa |
| `uniform_sampling` | `evox.operators.sampling` | NSGA3, RVEA, RVEAa, MOEAD |
| `clamp` | `evox.utils` | ALL |
| `randint` | `evox.utils` | RVEA, RVEAa |
| `nanmax` / `nanmin` | `evox.utils` | RVEA, RVEAa |
| `lexsort` | `evox.utils` | HypE |
| `minimum` | `evox.utils` | MOEAD |

## Common Algorithmic Pattern

All six algorithms share a core iteration pattern:

```
step():
  1. Mating selection: choose parents from current population
  2. Crossover: recombine selected parents
  3. Mutation: perturb offspring (bounded by lb/ub)
  4. Clamp: ensure offspring stay within bounds
  5. Evaluate: self.evaluate(offspring) → off_fit
  6. Merge: concatenate current and offspring populations/fitnesses
  7. Environmental selection: reduce merged set back to pop_size survivors
```

## Per-Algorithm Details

### NSGA2 (`nsga2.py`) — Dominance + Crowding Distance
- **Mating selection**: `tournament_selection_multifit` on `[-dis, rank]` (prefers low rank, then high crowding distance)
- **Environmental selection**: `nd_environmental_selection` returns survivors + `(rank, dis)` used in next generation's mating
- **State**: `pop`, `fit`, `rank`, `dis`
- **Operators**: simulated_binary, polynomial_mutation

### NSGA3 (`nsga3.py`) — Reference-Point Niching
- **Mating selection**: `tournament_selection_multifit` on `[rank]` only (no crowding distance)
- **Environmental selection**: Fully custom in-step — non-dominated ranking → ideal-point normalization → reference-point association → niche preservation via `torch.while_loop`
- **Reference points**: Generated via `uniform_sampling` (Das & Dennis method); stored as `self.ref`
- **State**: `pop`, `fit`, `rank`, `ref`
- **Complexity**: The most complex MO algorithm in this directory (~256 lines vs ~100–150 for others)
- **Helper functions**: `_compute_distances` (angle-penalized), plus several `vmap`-decorated inner functions for extreme-point finding, table row construction, and niche selection

### RVEA (`rvea.py`) — Reference Vector Guided
- **Mating selection**: Custom `_mating_pool()` — random selection from valid (non-NaN) individuals
- **Environmental selection**: `ref_vec_guided` using angle-penalized distance
- **Reference vectors**: From `uniform_sampling`; periodically adapted via `_rv_adaptation()` scaling by objective range
- **Parameters**: `alpha` (penalty rate, default 2.0), `fr` (adaptation frequency, default 0.1), `max_gen`
- **State**: `pop`, `fit`, `reference_vector`, `gen`, `init_v`
- **Key feature**: Reference vector adaptation every `1/fr` generations using `torch.cond`

### RVEAa (`rveaa.py`) — Adaptive RVEA with Regeneration
- **Extends RVEA** with reference vector regeneration (`_rv_regeneration`) — creates new vectors for unassociated regions
- **Pre-filtering**: Applies `non_dominate_rank` before `ref_vec_guided` selection (only rank-0 solutions participate)
- **Batch truncation**: At the final generation, halves the population via cosine-similarity-based truncation
- **Two vector sets**: `self.reference_vector` = cat(adapted initial vectors, regenerated vectors)
- **Compilation support**: Uses `torch.compiler.is_compiling()` guard to choose between `torch.cond` (compile-safe) and Python `if/else`

### MOEAD (`moead.py`) — Decomposition-Based
- **Core idea**: Weight vectors decompose MOP into `pop_size` scalar subproblems
- **Neighborhood**: Each subproblem has `n_neighbor = ceil(pop_size / 10)` neighbors (Euclidean distance on weight vectors)
- **Replacement**: Loop over subproblems; for each, select 2 random neighbors as parents, produce 1 offspring, and replace worse neighbors via PBI comparison
- **Scalarization**: PBI (Penalty-based Boundary Intersection): `d1 + 5 * d2`
- **Default crossover**: `simulated_binary_half` (not `simulated_binary` — produces 1 child, not 2)
- **State**: `pop`, `fit`, `z` (ideal point for scalarization)
- **Note**: Uses a Python `for` loop over `pop_size`; acknowledged as not fully tensorized (see source docstring)
- **No mating selection operator**: parent selection is neighbor-based, not tournament-based

### HypE (`hype.py`) — Hypervolume-Based
- **Core idea**: Use Monte Carlo hypervolume contribution as selection criterion
- **Hypervolume calculation**: `cal_hv()` — samples `n_sample` random points in objective space, counts dominated samples, uses inclusion-exclusion formula
- **Mating selection**: `tournament_selection` on `-hv` (higher hypervolume contribution = better, negated for minimization-style selection)
- **Environmental selection**: Non-dominated rank first, then hypervolume contribution as tiebreaker via `lexsort([-dis, rank])`
- **State**: `pop`, `fit`, `ref` (reference point for hypervolume, initialized at 1.2× max fitness)
- **Parameter**: `n_sample` (default 10000) — controls MC accuracy vs. cost

## Design Patterns & Conventions

- **Operator injection**: All algorithms accept optional `selection_op`, `mutation_op`, `crossover_op` in `__init__` and fall back to sensible defaults. This enables users to swap operators without subclassing.
- **Bounds handling**: All use 1D `lb`/`ub` tensors (same dtype/device) passed through `clamp()` after mutation.
- **Population init**: Random uniform within `[lb, ub]` (except NSGA3 with `data_type=bool` which thresholds at 0.5).
- **Fitness init**: `torch.inf`-filled tensors for `fit` state.
- **NSGA3 exception**: Handles binary decision variables via `data_type` parameter.
- **RVEA/RVEAa exception**: Handle NaN populations from pre-filtering via masked mating pool selection.

## Constraints
- All algorithms are pure PyTorch — no NumPy, no CPU-only loops (MOEAD's Python `for` loop is a known exception, noted in its docstring).
- State must use `Mutable`; hyperparameters must use `Parameter`.
- `step()` must be compatible with `torch.compile` — use `torch.cond`/`torch.while_loop` for control flow depending on tensor values.
- `self.evaluate()` is injected externally; algorithms never import or call `Problem` directly.
