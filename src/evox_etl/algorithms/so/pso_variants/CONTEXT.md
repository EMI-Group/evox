# evox_etl/algorithms/so/pso_variants — functional PSO variants

## Intent
1:1 functional ports of the torch PSO-variant algorithms (read-only reference
in `../../../../evox/algorithms/so/pso_variants/`) on the STEP protocol
(`../../../../evox_etl/core/algorithm.py`): frozen config dataclasses,
tensor state dataclasses, and plain module-level `init`/`init_step`/`step`
functions. See `../../../DESIGN.md` §4-5.
All 7 variants are ported: `pso`, `cso`, `clpso`, `dms_pso_el`, `sl_pso_gs`,
`sl_pso_us`, `fs_pso` (+ shared `utils.py`), with the package `__init__.py`
mirroring the torch `__all__`.

## API Surface
- Every algorithm module exposes a frozen config dataclass, a `*State`
  dataclass, plain `init(config, key) -> state` and
  `step(config, state, evaluate) -> state` functions, and a module-level
  `make_*` functional constructor (`make_pso`, `make_cso`,
  `make_clpso`, `make_dms_pso_el`, `make_sl_pso_gs`, `make_sl_pso_us`,
  `make_fs_pso`), all exported from the package `__init__.py`.
- All 7 modules also define `init_step(config, state, evaluate) -> state`
  (their torch counterparts have `init_step`): evaluate the initial
  population and seed the best-trackers. No module defines `final_step`
  (torch has no PSO-level override; the workflow falls back to `step`).
- Step fusion: `step` = candidate generation (RNG draws,
  best-tracker updates via `replace(...)`) → `fitness = evaluate(pop)` →
  state update, all inside one trace. `evaluate(candidates) -> fitness` is the
  workflow-injected opaque traced closure (solution_transform → problem →
  opt-direction scaling → fitness_transform → monitor update; minimization
  semantics). It is never stored in the state or re-threaded.
- The array-like API (numpy lb/ub — and optional mean/stdev for `CSO`/
  `FSPSO` — exactly like torch) lives on the `make_*` constructors; the
  config dataclasses themselves store only plain static leaves (see below).
- **Config construction pattern (all 7 configs)**: configs are dumb frozen
  dataclasses with `lb`/`ub` (and cso/fs_pso `mean`/`stdev`) stored as flat
  float tuples (`tuple[float, ...]`) — plain static pytree leaves that pass
  `etl.build`/`etl.run` unchanged, no pytree registration (mo/ configs keep
  ndarrays with zero-child pytree registration, see `../mo/nsga2.py`).
  `make_*` is the single place that normalizes array-like input and
  validates (see "Config-constructor notes" below). Direct dataclass
  construction with already-normalized statics remains legal (it bypasses
  `make_*` validation).
- `utils.py` — `min_by` (concat axis 0, `etl.argmin(keys, axis=0)`, gather
  with reshaped (1,) index, reshape back to `x.shape[1:]`) and
  `random_select_from_mask` (noise + argsort + scatter-ones, key-first RNG).
- Clamp comes from `evox_etl.operators.jit_fix_operator` (`clamp`,
  `clamp_float`, `clamp_int`). Bound/stats constants are baked as (1, dim)
  float32 rows via the shared helpers `bake_bounds(lb, ub, as_row=True)` /
  `bake_float32_constant(values, shape=(1, -1))` from `../../_config_utils.py`;
  the module-local `_bounds`/`_bake`/`_bake_bounds` privates delegate to them,
  and numpy imports remain only in cso/fs_pso/dms_pso_el.
- Tests: `../../../../unit_test/etl/algorithms/so/pso_variants/` (the
  shared `helpers.run_generations` driver predates the step protocol and is
  rewritten in a later wave; until then these tests fail against these
  modules) + `../../../../unit_test/etl/algorithms/parity/test_pso_parity.py`.

## Step-protocol semantics per variant
- `pso`/`clpso`/`fs_pso`/`sl_pso_gs`/`sl_pso_us`: the fused `step` generates
  the proposed population, calls `evaluate(pop)` once on it, then applies
  the state update (`fit=fitness`; cso scatters
  onto student rows; dms also counts the generation). All randomness is
  drawn BEFORE `evaluate`, so `step` is
  deterministic given the state.
- `cso` evaluates only the STUDENT rows (candidates of shape
  `(pop_size // 2, dim)`) and scatters fitness onto the students' rows; the
  `students` index tensor stays in the state (monitor sees the raw
  candidates, torch parity).
- `dms_pso_el` keeps its internal 0-d int32 `iteration` counter: strategy
  dispatch (`iteration < 0.9 * max_iteration`) and the regroup cadence
  (`iteration % regrouped_iteration_num`) run inside `step`, and the counter
  advances AFTER the evaluate/record write (torch increments before its
  evaluate; the fused order yields the identical net state sequence —
  verified bit-identical against the pre-conversion baseline).
- The generation-0 candidate/record pair fuses into `init_step` =
  evaluate-the-initial-population + the first-generation record body,
  matching each torch `init_step` 1:1 (note the torch quirks preserved:
  clpso/fs_pso/sl_pso_* set `global_best_fit = min(fitness)` but do NOT
  update `global_best_location`).

## Verified ETL facts (do not re-investigate)
- np.ndarray config fields ARE accepted as static trace values by the
  installed etl (see `../../CONTEXT.md` "ETL issues found" #1) — direct
  ndarray config construction still graphs, but it bypasses `make_*`
  normalization/validation. Plain float-tuple storage stays the norm
  (frozen `__eq__`/`__hash__` on tuples, static-legal leaves, normalized
  once by `make_*`).
- Newaxis indexing (`x[None, :]`) raises TraceError — use `enp.expand_dims`.
- `etl.min(x, axes=0)` on (n,) returns a scalar; `etl.argmin(x, axis=0)`
  returns an int64 scalar.
- `random.split_n(key, n)` returns n keys; `random.randint(key, (n,), low,
  high, dtype=etl.int32)`; `random.uniform/normal(key, shape, ..., etl.float32)`.
- `etl.select(cond, a, 0)` broadcasts python scalars; int32/int64 1-D indices
  work with `etl.gather(x, idx, axis=0)` (numpy-take semantics).
- `enp.zeros((n,), dtype=etl.float32)` and `enp.full((), float("inf"),
  dtype=etl.float32)` work inside traces.
- Draws are in torch order, one subkey each: rg, rp, tournament1, tournament2,
  offset, mutation_prob (then store the advanced key back in the state).
- torch `step` concatenates 2*half rows, so odd `pop_size` shrinks to
  2*(pop_size//2) after the first step — ported as-is (torch parity).
- The workflow-owned `evaluate` closure is called INSIDE the traced step
  body (nonlocal problem/monitor state threading works — validated in
  `core/workflow.py` `_make_step_fn`); configs reach etl by closure capture
  in the workflow, or as STATIC args to both `etl.build` and `etl.run` in
  direct drivers.

## Config-constructor notes (current state)
- All normalization/validation lives in the module-level `make_*`
  constructors. Policy is binding in `../../../DESIGN.md` §4.1;
  normalization/baking helpers are consolidated in
  `evox_etl.algorithms._config_utils` (`normalize_bounds`, `to_float_tuple`,
  `bake_bounds`, `bake_float32_constant`).
- `make_*` validation raises `ValueError` with clear messages (no bare
  asserts): lb/ub must be 1-D and length-matching; cso/fs_pso optional
  mean/stdev are validated when given (1-D, length == dim); dms_pso_el
  bounds must be 1-D (torch requires 1-D too). Numerics are unchanged for
  valid inputs.
- Field-read sites tolerate both tuple configs and legacy ndarray direct
  constructions: use `len(config.lb)` for dim (prefer over `.shape[0]`);
  `np.asarray(config.lb, dtype=np.float32)` still works for host-side baking.

## Routing Table
| Area | Path |
|---|---|
| Package init (`__all__` + submodule imports incl. `make_*`) | `__init__.py` |
| PSO variants (all 7) | `pso.py`, `cso.py`, `clpso.py`, `dms_pso_el.py`, `sl_pso_gs.py`, `sl_pso_us.py`, `fs_pso.py` |
| Shared helpers (min_by, random_select_from_mask) | `utils.py` |
| Config helpers (normalize_bounds, to_float_tuple, bake_bounds, bake_float32_constant) | `../../_config_utils.py` (parent module — shared by all algorithm families) |
| Tests | `../../../../unit_test/etl/algorithms/so/pso_variants/` (sibling tree) |
| Parity tests | `../../../../unit_test/etl/algorithms/parity/` (sibling tree) |
| Torch reference (READ-ONLY) | `../../../../evox/algorithms/so/pso_variants/` |
| Operator/util helpers | `../../../operators/jit_fix_operator.py` |
| Step-protocol contract | `../../../../evox_etl/core/algorithm.py` + `core/workflow.py` |
