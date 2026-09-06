# evox_etl/algorithms/so/pso_variants — functional PSO variants

## Intent
1:1 functional ports of the torch PSO-variant algorithms (read-only reference
in `../../../../evox/algorithms/so/pso_variants/`) as `init/ask/tell` plain
functions + frozen config/state dataclasses. See `../../../DESIGN.md` §4-5.
All 7 variants are ported: `pso`, `cso`, `clpso`, `dms_pso_el`, `sl_pso_gs`,
`sl_pso_us`, `fs_pso` (+ shared `utils.py`), with the package `__init__.py`
mirroring the torch `__all__`.

## API Surface
- Every algorithm module exposes a frozen config dataclass, a `*State`
  dataclass, `init/init_ask/init_tell/ask/tell` plain functions, and a
  module-level `make_*` functional constructor (`make_pso`, `make_cso`,
  `make_clpso`, `make_dms_pso_el`, `make_sl_pso_gs`, `make_sl_pso_us`,
  `make_fs_pso`), all exported from the package `__init__.py`.
- The array-like API (numpy lb/ub — and optional mean/stdev for `CSO`/
  `FSPSO` — exactly like torch) lives on the `make_*` constructors; the
  config dataclasses themselves store only plain static leaves (see below).
- **Config construction pattern (all 7 configs)**: configs are dumb frozen
  dataclasses with `lb`/`ub` (and cso/fs_pso `mean`/`stdev`) stored as flat
  float tuples (`tuple[float, ...]`) — plain static pytree leaves that pass
  `etl.build`/`etl.run` unchanged, no pytree registration (mo/ configs keep
  ndarrays with zero-child pytree registration, see `../mo/nsga2.py`).
  `__post_init__` normalizers were deleted; `make_*` is now the single place
  that normalizes array-like input and validates (see "Config-constructor
  notes" below). Direct dataclass construction with already-normalized
  statics remains legal (it bypasses `make_*` validation).
- `utils.py` — `min_by` (concat axis 0, `etl.argmin(keys, axis=0)`, gather
  with reshaped (1,) index, reshape back to `x.shape[1:]`) and
  `random_select_from_mask` (noise + argsort + scatter-ones, key-first RNG).
- Clamp comes from `evox_etl.operators.jit_fix_operator` (`clamp`,
  `clamp_float`, `clamp_int`). Bound/stats constants are baked as (1, dim)
  float32 rows via the shared helpers `bake_bounds(lb, ub, as_row=True)` /
  `bake_float32_constant(values, shape=(1, -1))` from `../../_config_utils.py`;
  the module-local `_bounds`/`_bake`/`_bake_bounds` privates delegate to them,
  and numpy imports remain only in cso/fs_pso/dms_pso_el.
- Tests: `../../../../unit_test/etl/algorithms/so/pso_variants/` (7 etl-only
  smoke tests) + `../../../../unit_test/etl/algorithms/parity/
  test_pso_parity.py` (torch-vs-etl PSO parity on Sphere); all green.

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

## Config-constructor notes (current state)
- `__post_init__` normalizers were removed in the config-constructor refactor:
  all 7 configs are dumb frozen dataclasses storing flat float tuples, and
  the module-level `make_*` constructors are the single place that normalizes
  array-like input and validates. Policy is binding in `../../../DESIGN.md`
  §4.1; normalization/baking helpers are consolidated in
  `evox_etl.algorithms._config_utils` (`normalize_bounds`, `to_float_tuple`,
  `bake_bounds`, `bake_float32_constant`).
- `make_*` validation raises `ValueError` with clear messages (no bare
  asserts): lb/ub must be 1-D and length-matching; cso/fs_pso optional
  mean/stdev are validated when given (1-D, length == dim); dms_pso_el
  bounds must be 1-D (the old silent `.ravel()` was removed — torch requires
  1-D too). Numerics are unchanged for valid inputs.
- Configs reach etl by two routes: unit tests pass the config as a STATIC arg
  to both `etl.build` and `etl.run` (`unit_test/etl/algorithms/helpers.py`
  `run_generations`, re-validated by value each run), while
  `core/workflow.py` `StdWorkflow` closure-captures configs inside the traced
  body (never a build/run arg).
- Existing unit tests still direct-construct configs with np.ndarray kwargs
  (graph-valid on installed etl, but they bypass `make_*` validation);
  migrating them to `make_*` is a separate test-migration wave — do NOT edit
  `../unit_test/etl` from here.
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
