# evox_etl/problems/numerical — benchmark problems in functional style

## Intent
Functional-ETL port of the numerical problems from torch evox
(`../../../evox/problems/numerical/`, read-only): `basic.py` (done), `dtlz.py`
and `cec2022.py` (written by other agents in parallel). Each problem is a
frozen config dataclass + a plain module-level `evaluate(config, problem_state,
pop) -> (fitness, problem_state)` function — no `@etl.defn` decorators (§4.3).

## API Surface
- `basic.py`: 10 configs (`ShiftAffineNumericalProblem`, `Ackley`, `Griewank`,
  `Rastrigin`, `Rosenbrock`, `Schwefel`, `Sphere`, `Ellipsoid`, `Zakharov`,
  `Levy`) + 9 funcs (`*_func`, torch signatures) + module-level `evaluate`.
- `state.py`: `ProblemState` — empty frozen dataclass (numerical problems are
  stateless; the signature is kept uniform).
- `__init__.py`: re-exports all of the above + `CEC2022`/`DTLZ1-7` from the
  sibling modules.

## Constraints / Notes for Agents
- Shift/affine numpy arrays in configs are baked INSIDE `evaluate` as graph
  constants: `etl.constant(etl.tensor(np.asarray(config.shift,
  dtype=np.float32)))` — never closure-captured concrete tensors. When shift
  is None, add a zero constant instead (static Python branch on config).
- Config classes are registered as zero-child pytree nodes
  (`etl.register_pytree_node`) so `etl.build(evaluate, config, ...)` passes the
  whole config through as opaque static data — numpy arrays are NOT valid etl
  trace inputs by themselves. Follow this pattern in dtlz.py/cec2022.py.
- Partial-axis slicing over a symbolic batch dim fails in etl's IR slice op
  (e.g. `x[:, 1:]`) — use `etl.gather` with static int32 `enp.arange` indices.
- numpy IS used in basic.py (config array handling: `np.asarray`/`np.zeros`)
  — this is the §4.1 config convention and supersedes the general no-numpy
  rule.
- cec2022 input data lives in `../../../evox/problems/numerical/cec2022_input_data/`.
- DTLZ/ZDT suites expose `pf()` reference-front generators (used by
  metrics/tests).

## Validation
Throwaway script pattern (not committed): load basic.py/state.py directly via
`importlib.util.spec_from_file_location` (the package `__init__` imports
dtlz/cec2022 which may not exist yet), `etl.build(evaluate, config,
ProblemState(), TensorSpec((None, dim), float32))` + `etl.run(exe, config,
ProblemState(), x)`. Verified: all 9 functions bit-exact vs the torch
formulas (numpy reference), shift/affine variants match torch evox within
float32 rounding, torch assert messages preserved.
