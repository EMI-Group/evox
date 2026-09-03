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
  `tell(config, state, fitness) -> state`. `__init__.py` mirrors torch exports.
- State leaves are etl tensors ONLY (float32 preferred, int32/int64 indices, key `()`
  int64). `tell` never consumes randomness; `ask` advances `state.key` when it draws.
- ODE runs a two-phase state machine: one torch ODE step = 2 etl generations
  (phase 0 = DE mutation/crossover; phase 1 = opposition selection on the
  post-selection population) — the fixed one-evaluate-per-generation contract makes
  a single ask/tell pair impossible for the second evaluate.

## Constraints
- Import every operator/util function from `evox_etl.algorithms._operator_shims`
  (clamp, DE_* crossovers, select_rand_pbest, ...) and `_take_along_axis` from
  `evox_etl.algorithms._shim_selection_basic` — NOT from `evox_etl.operators`
  (parallel task, may not exist yet).
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

## Status
- ALL SIX modules implemented: `de.py` (`DE`), `ode.py` (`ODE`, two-phase state
  machine — one torch ODE step = 2 etl generations), `jade.py` (`JaDE`),
  `shade.py` (`SHADE`), `sade.py` (`SaDE`), `code.py` (`CoDE`). `__init__.py`
  mirrors the torch exports: `DE, CoDE, JaDE, ODE, SaDE, SHADE`.
- Smoke tests: `../../../../unit_test/etl/algorithms/so/de_variants/` (de, ode
  green). Parity test: `../../../../unit_test/etl/algorithms/parity/test_de.py`.

## Known issues
- The DE parity test (torch vs etl, Sphere 40-dim, pop 100, 20 gens, seed 0,
  margin `etl_best <= torch_best * 1.1 + 1e-3`) FAILS as written: torch best
  27156.33 vs etl best 43143.73 (threshold 29871.97). Cause is stochastic RNG
  noise, NOT a port bug — etl RNGs (Threefry/SplitMix/Philox) can never match
  torch's MT19937 bit-for-bit, and a 60-seed paired sweep showed no systematic
  bias (mean diff 1601 ± 7727, n.s.; P(etl > torch) ≈ 0.53; P(fail margin) ≈
  0.37). Pending root-agent decision (different seed / looser margin / extra
  generations).

## Routing Table
| Area | Path |
|---|---|
| DE + ODE (shared mutation/crossover helpers) | `de.py`, `ode.py` |
| JaDE / SHADE / SaDE / CoDE | `jade.py`, `shade.py`, `sade.py`, `code.py` |
| Smoke tests (no torch) | `../../../../unit_test/etl/algorithms/so/de_variants/` (sibling — write via bash heredoc) |
| Parity test (torch allowed) | `../../../../unit_test/etl/algorithms/parity/test_de.py` (sibling) |
| Operator shims (import from) | `../../_operator_shims.py`, `../../_shim_selection_basic.py` |
