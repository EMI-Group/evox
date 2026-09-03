# evox_etl/problems — benchmark problems in functional style

## Intent
Port of the numerical problems from torch evox (`../../../evox/problems/`, read-only):
basic.py (Sphere, Rastrigin, Rosenbrock, Ackley, Griewank, Schwefel, Zakharov,
Levy...), dtlz.py (DTLZ1-7 + helpers), cec2022.py (with shipped input data). Plain
functions in etl style — no classes, no OOP; the workflow traces them (per
`../DESIGN.md` §4.3, functions are NOT `@etl.defn`-wrapped).

NOT ported (external-library dependent — reported to root): neuroevolution problems
(brax, mujoco_playground, supervised_learning, virtual_lora), hpo_wrapper.

## API Surface
- **Uniform problem signature**: `evaluate(config, problem_state, pop) ->
  (fitness, problem_state)` — plain function, `pop` shape `(n, dim)`, fitness
  `(n,)` (SO) or `(n, n_obj)` (MO). All problems minimize internally; the
  workflow applies `opt_direction` for maximization.
- **Configs** are frozen dataclasses mirroring the torch class `__init__`
  signatures; boundary/shift/affine tensor fields become numpy arrays in the
  config (baked once as graph constants inside functions, per DESIGN §4.1).
- **Statelessness**: numerical problems are stateless and all share the empty
  frozen `ProblemState` (defined in `numerical/state.py`, re-exported at
  package level). The `(fitness, problem_state)` signature is kept for
  uniformity with stateful problems.
- **DTLZ reference fronts**: DTLZ1-7 expose `pf(config)` functions (plain
  functions meant to be traced) used by metrics/tests. They use private local
  `_uniform_sampling`/`_grid_sampling` helpers inside dtlz.py (the operators
  milestone will dedupe these into `operators/`).
- **No boundary handling in basic problems**: raw math only, exactly like torch
  evox (constraining is the algorithm's responsibility there).
- Package `__init__` re-exports the torch `evox.problems.numerical` export
  surface plus `Zakharov`, `Levy`, `zakharov_func`, `levy_func` (torch basic.py
  has them but doesn't re-export) and `ProblemState`.

## Notes for Agents
- cec2022 input data lives in `../../../evox/problems/numerical/cec2022_input_data/`
  — numpy allowed ONLY for loading that data; bake it as constant tensors (closure).

## Routing Table
| Area | Path | Notes |
|---|---|---|
| Numerical problems (basic, dtlz, cec2022, state) | `numerical/` | single subpackage; owns `ProblemState` |
| Tests | `../../../unit_test/etl/problems/` | sibling — mirrors this package |
| Reference (torch) impl | `../../../evox/problems/` | sibling — READ-ONLY, never modify |
