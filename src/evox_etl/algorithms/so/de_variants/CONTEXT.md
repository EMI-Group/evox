# evox_etl/algorithms/so/de_variants — functional DE-family ports

## Intent
Plain-function `init(config, key)` / `ask(config, state)` / `tell(config, state, fitness)`
ports of the torch evox DE family (READ-ONLY reference: `../../../../evox/algorithms/
so/de_variants/`), one module per torch file: `de.py`, `jade.py`, `shade.py`, `sade.py`,
`ode.py`, `code.py`. Binding spec: `../../DESIGN.md` (read fully before writing code).

## API Surface
- Each module: frozen config dataclass named after the torch class (`DE`, `JaDE`,
  `SHADE`, `SaDE`, `ODE`, `CoDE`) + frozen `<Name>State` dataclass + plain functions
  `init(config, key) -> State`, `ask(config, state) -> (candidates, state)`,
  `tell(config, state, fitness) -> state`.
- Each config module also defines a `make_*` constructor in the SAME module
  (`make_de`, `make_jade`, `make_shade`, `make_sade`, `make_ode`, `make_code`);
  `__init__.py` exports the classes and all 6 `make_*`.
- `init_ask`/`init_tell` encode torch's `init_step` (evaluate the initial
  population) on `de.py`, `ode.py`, `jade.py` — the torch counterparts of
  SHADE/SaDE/CoDE have no init_step, so those modules omit the pair.
- State leaves are etl tensors ONLY (float32 preferred, int32/int64 indices, key `()`
  int64). `tell` never consumes randomness; `ask` advances `state.key` when it draws.
- ODE runs a two-phase state machine: one torch ODE step = 2 etl generations
  (phase 0 = DE mutation/crossover; phase 1 = opposition selection on the
  post-selection population) — the fixed one-evaluate-per-generation contract makes
  a single ask/tell pair impossible for the second evaluate.

## Constraints
- Import crossover/sampling/selection operators from `evox_etl.operators`
  (canonical, torch-parity-verified) and the jit-fix utils (clamp, clamp_float,
  `_take_along_axis`, ...) from `evox_etl.operators.jit_fix_operator`.
- No torch/numpy in graph code (numpy only for baking constants via
  `etl.ops.constant(etl.core.tensor(np.asarray(...)))`).
- Files < ~400 lines; static Python loops/config branches allowed in traces.

## Notes for agents (verified — do not re-investigate)
- **np.ndarray config fields are REJECTED by `etl.build`** ("neither a TensorSpec nor
  a static Python value"). Normalize `lb`/`ub`/`mean`/`stdev`/`differential_weight`/
  `param_pool` to tuples of plain Python floats via `object.__setattr__` in
  `__post_init__` (callers may pass np arrays; stored values are tuples).
- **`etl.select` does NOT numpy-broadcast** a `(n,)` condition against `(n, m)`
  operands (ShapeError). Always `enp.expand_dims(cond, 1)` first. Scalar conds
  broadcast fine. Python float + float32 → float32 OK; Python int + int32 → int64
  (avoid — use `enp.full/ones/zeros/arange` with explicit dtype).
- `etl.gather` = numpy take semantics; a 0-d index drops the indexed axis
  (result `(dim,)`). Row-local gather needs `_take_along_axis` (2-D only).
- RNG: `key, subkey = random.split(state.key)` at function entry; one split per
  random op, in torch's draw order; store the advanced `key` back in the state.
  `random.multinomial(key, probs, n)` returns int32.
- Bounds baked ONCE per function as `(1, dim)` constants via `etl.ops.constant`;
  `dim = len(config.lb)` stays a static Python int.
- SHADE/SaDE/CoDE init populations use **`randn` scaled by bounds** (torch quirk —
  no uniform, no clamp) — do not assert in-bounds on their initial pops.
- `etl.median(x, axis=0)` matches torch's NaN propagation; `etl.roll(x, shift, axis)`;
  `etl.nansum(x, axes=0)`; `etl.argsort(x, axis=0, stable=True)` (int64 indices).
- Verified host pattern (dataclass config as static arg + dataclass state pytree):
  `etl.build(fn, cfg, spec_tree, backend="numpy")` / `etl.run(exe, cfg, state)` —
  tuple outputs come back as tuples. See `unit_test/etl/algorithms/helpers.py`
  `run_generations` for the full loop.
- Parity tests MUST use a unique basename (`test_<name>_parity.py`, not
  `test_<name>.py`): pytest's prepend import mode collides when the same basename
  exists in `so/de_variants/` and `parity/` and both are collected in one run.
- The DE parity test uses the MEDIAN of 3 seeds with the 10% margin: per-seed
  best fitnesses fluctuate ~0.57-1.92× around torch (different RNG streams), so a
  single-seed 10% margin fails ~1/3 of the time by chance. Median-of-3 is stable.
- **ode.py imports de.py private helpers** (`_de_trial`, `_bounds`, `_constant_1d`,
  `_to_float_tuple`, ode.py:29-34) — de.py internals are part of ODE's API; any
  de.py refactor (e.g. moving __post_init__ normalization into a builder) must
  keep ode.py compiling or update the import.
- All six config classes are constructed KEYWORD-ONLY with `lb`/`ub` as
  np.ndarray at every in-repo call site (smoke tests test_{de,ode,jade,shade,
  sade,code}.py, parity test_de_parity.py:45, benchmarks/etl_vs_torch/bench_so.py:115);
  `pop_size`/`lb`/`ub` are required (no defaults), so default-construction is
  impossible. Tests only exercise default hyperparameters — the tuple
  `differential_weight` (ndv>1), `mean`/`stdev` init, and non-default
  `param_pool` normalization paths have NO test coverage.

## Routing Table
| Area | Path |
|---|---|
| DE + ODE (shared mutation/crossover helpers) | `de.py`, `ode.py` |
| JaDE / SHADE / SaDE / CoDE | `jade.py`, `shade.py`, `sade.py`, `code.py` |
| Smoke tests (no torch) | `../../../../unit_test/etl/algorithms/so/de_variants/` (sibling — write via bash heredoc) |
| Parity test (torch allowed) | `../../../../unit_test/etl/algorithms/parity/test_de_parity.py` (sibling) |
| Canonical operators (import from) | `../../../operators/crossover`, `../../../operators/selection`, `../../../operators/jit_fix_operator.py` |
