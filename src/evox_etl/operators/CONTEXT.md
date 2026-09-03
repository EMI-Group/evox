# evox_etl/operators — pure functional genetic operators

## Intent
Port of torch evox operators (`../../../evox/operators/`, read-only) as PURE functions:
`fn(config, key, *tensors) -> tensor`. No defn decorator needed (called inside traces).
Config = frozen dataclass of hyperparameters. See `../../DESIGN.md` §4.3.

## Routing Table
| Area | Path | Files |
|---|---|---|
| Sampling | `sampling/` | uniform.py (Das-Dennis), latin_hypercube.py, gird.py (grid) |
| Selection | `selection/` | non_dominate.py (non_dominated_sort + crowding_distance), tournament_selection.py, find_pbest.py, rvea_selection.py |
| Crossover | `crossover/` | differential_evolution.py, sbx.py, sbx_half.py |
| Mutation | `mutation/` | pm_mutation.py |
