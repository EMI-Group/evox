# evox_etl/algorithms/so/pso_variants — functional PSO variants

## Intent
1:1 functional ports of the torch PSO-variant algorithms (read-only reference
in `../../../../evox/algorithms/so/pso_variants/`) as `init/ask/tell` plain
functions + frozen config/state dataclasses. See `../../../DESIGN.md` §4-5.
All 7 variants are ported: `pso`, `cso`, `clpso`, `dms_pso_el`, `sl_pso_gs`,
`sl_pso_us`, `fs_pso` (+ shared `utils.py`), with the package `__init__.py`
mirroring the torch `__all__`.

## API Surface
- Every algorithm module exposes its config dataclass (constructor API takes
  numpy lb/ub — and mean/stdev for `CSO`/`FSPSO` — exactly like torch), a
  `*State` dataclass, and `init/init_ask/init_tell/ask/tell` plain functions.
- **Config pattern (uniform across all 7)**: each frozen config's
  `__post_init__` validates `lb`/`ub` on `np.asarray` copies (ndim==1, shapes
  match) and then normalizes them to float tuples via `object.__setattr__`
  (plus `mean`/`stdev` when not None for `CSO`/`FSPSO`). etl's tracer accepts
  tuple-of-float leaves as static values, so configs pass
  `etl.build`/`etl.run` unchanged — no pytree registration. Type annotations
  stay `np.ndarray` / `np.ndarray | None`. Read `dim` via `len(config.lb)`
  (tuples have no `.shape`); `np.asarray(config.lb, dtype=np.float32)` still
  works for constant baking.
- `utils.py` — `min_by` (concat axis 0, `etl.argmin(keys, axis=0)`, gather
  with reshaped (1,) index, reshape back to `x.shape[1:]`) and
  `random_select_from_mask` (noise + argsort + scatter-ones, key-first RNG).
- Clamp comes from `evox_etl.algorithms._jit_fix_operator` (`clamp`,
  `clamp_float`, `clamp_int`), bounds are baked as (1, dim) float32 constants
  via `etl.ops.constant(core.tensor(np.asarray(...)[None, :]))` inside each
  function needing them (numpy only for constant baking).
- Tests: `../../../../unit_test/etl/algorithms/so/pso_variants/` (7 etl-only
  smoke tests) + `../../../../unit_test/etl/algorithms/parity/
  test_pso_parity.py` (torch-vs-etl PSO parity on Sphere); all green.

## Verified ETL facts (do not re-investigate)
- **numpy arrays are NOT legal static trace inputs**: etl's `_is_static_value`
  accepts only None/bool/int/float/complex/str/Enum/dtype/slice/Dim/DimExpr/
  Device. Passing a config with numpy lb/ub to `etl.build`/`etl.run` raises
  TraceError. Solution: normalize array fields to float tuples in
  `__post_init__` (see above) — no `etl.register_pytree_node` needed.
  (Root-cause note: the DESIGN.md §4.3 "no partialization needed" claim only
  holds for scalar-leaf configs; the helpers' `run_generations` could
  alternatively closure-capture configs via `functools.partial`.)
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

## Routing Table
| Area | Path |
|---|---|
| Package init (`__all__` + submodule imports) | `__init__.py` |
| PSO variants (all 7) | `pso.py`, `cso.py`, `clpso.py`, `dms_pso_el.py`, `sl_pso_gs.py`, `sl_pso_us.py`, `fs_pso.py` |
| Shared helpers (min_by, random_select_from_mask) | `utils.py` |
| Tests | `../../../../unit_test/etl/algorithms/so/pso_variants/` (sibling tree) |
| Parity tests | `../../../../unit_test/etl/algorithms/parity/` (sibling tree) |
| Torch reference (READ-ONLY) | `../../../../evox/algorithms/so/pso_variants/` |
| Operator/util helpers | `../../_jit_fix_operator.py` |
