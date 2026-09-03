# evox_etl/operators — pure functional genetic operators

## Intent
Port of torch evox operators (`../../../evox/operators/`, read-only) as PURE functions.
The torch operators are ALREADY pure functions — porting rule (binding, see
`../../DESIGN.md` §4.3):
- Keep torch function names, argument names and order EXACTLY.
- Functions using randomness in torch gain `key` as the FIRST parameter
  (etl.random; split from the caller's key).
- Drop the `device: torch.device` parameter (`latin_hypercube_sampling_standard`).
- `torch.Tensor` → etl tensors; translate ops 1:1 (where→select, clamp→clamp,
  argsort→argsort, gather→gather, cumsum→cumsum, rand→random.uniform etc.).

## Routing Table
| Area | Path | Files |
|---|---|---|
| Sampling | `sampling/` | uniform.py (Das-Dennis), latin_hypercube.py, gird.py (grid) |
| Selection | `selection/` | non_dominate.py (dominate_relation, non_dominate_rank, crowding_distance, nd_environmental_selection), tournament_selection.py, find_pbest.py, rvea_selection.py |
| Crossover | `crossover/` | differential_evolution.py, sbx.py, sbx_half.py |
| Mutation | `mutation/` | pm_mutation.py |
