# Selection Operators

## Intent
Provides selection operators for evolutionary computation, covering both **multi-objective** (Pareto-based) and **single-objective** (fitness-based) paradigms. All operators are pure functions operating on PyTorch tensors — stateless, no `ModuleBase` inheritance.

## API Surface

### Multi-Objective Selection (Pareto)
- **`non_dominate_rank(f)`** — Computes non-dominated Pareto ranks for a population given objective values `f: (n, m)`. Returns rank tensor `(n,)`.
- **`crowding_distance(costs, mask)`** — Computes crowding distance (diversity metric) within a Pareto front subset identified by a boolean `mask`. Returns distance tensor `(n,)`.
- **`nd_environmental_selection(x, f, topk)`** — Full environmental selection: applies `non_dominate_rank` + `crowding_distance`, then lexicographically sorts (rank ascending, crowding descending) and returns the top `k` solutions with their objectives, ranks, and distances.

### Single-Objective Selection
- **`tournament_selection(n_round, fitness, tournament_size=2)`** — Standard tournament selection minimizing a 1D fitness tensor. Returns indices of winners.
- **`select_rand_pbest(percent, population, fitness)`** — Selects a uniformly random individual from the top `percent` fraction of the population (personal-best pool), per each individual. Used in DE/PSO variants like SHADE.

### Multi-Criteria / Decomposition Selection
- **`tournament_selection_multifit(n_round, fitnesses, tournament_size=2)`** — Tournament selection across multiple fitness criteria using lexicographic ordering (`lexsort`). Receives a list of 1D tensors, one per objective.
- **`ref_vec_guided(x, f, v, theta)`** — RVEA selection: assigns solutions to reference vectors by minimum angle, then selects per-vector via Angle-Penalized Distance (APD). Returns `(next_x, next_f)`.

## Internal Helpers (not exported)
- **`dominate_relation(x, y)`** — Constructs a domination matrix `A` where `A[i,j] = True` if `x_i` dominates `y_j`.
- **`update_dc_and_rank`** — Single Pareto-front peeling step: marks current front, updates dominate counts.
- **`_iterative_get_ranks`** — Iterative non-dominated sorting loop with `torch.compile` and vmap support via `register_vmap_op`.
- **`apd_fn`** — Angle-Penalized Distance calculation for RVEA.

## Constraints
- All inputs are PyTorch tensors on GPU or CPU.
- `non_dominate_rank` and `nd_environmental_selection` support `torch.compile` and vmap batching (registered via `evox.utils.register_vmap_op`).
- Tournament selection assumes **minimization** (`torch.argmin`).
- `ref_vec_guided` currently uses a suboptimal per-vector argmin; a future CUDA `segment_sort`/`segment_argmin` is planned (see docstring note).

## Dependencies
- `torch` — Tensor operations, `torch.compile`, `torch.while_loop`
- `evox.utils` — `lexsort`, `register_vmap_op`, `clamp_float`, `maximum`, `nanmin`

## Routing Table
- `non_dominate.py` → Pareto ranking, crowding distance, environmental selection for multi-objective optimization
- `tournament_selection.py` → Single-fitness and multi-fitness tournament selection
- `find_pbest.py` → Random personal-best selection for DE/PSO
- `rvea_selection.py` → RVEA reference-vector-guided selection
