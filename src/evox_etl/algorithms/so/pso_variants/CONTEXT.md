# evox_etl/algorithms/so/pso_variants — functional PSO variants

## Intent
1:1 functional ports of the torch PSO-variant algorithms (read-only reference
in `../../../../evox/algorithms/so/pso_variants/`) as `init/ask/tell` plain
functions + frozen config/state dataclasses. See `../../../DESIGN.md` §4-5.

## API Surface
- `fs_pso.py` — `FSPSO` config (numpy lb/ub/mean/stdev fields, same defaults as
  torch), `FSPSOState`, `init/init_ask/init_tell/ask/tell`. Ports the torch
  fs_pso step 1:1 (elite selection by argsort, PSO velocity update on the
  elite half, tournament crossover + mutation for the offspring half).
  `init_tell` does NOT update `global_best_location` — torch parity.
- `utils.py` — TEMPORARY minimal `min_by` (concat axis 0, `etl.argmin(keys,
  axis=0)`, gather with reshaped (1,) index, reshape back to `x.shape[1:]`).
  A parallel agent writes the full utils.py; this copy only exists so fs_pso
  is self-contained. Manager resolves any merge conflict.
- Clamp comes from `evox_etl.algorithms._operator_shims` (`clamp`), bounds are
  baked as (1, dim) float32 constants via
  `etl.ops.constant(core.tensor(np.asarray(...)[None, :]))` inside each
  function needing them (numpy only for constant baking).
- Smoke test: `../../../../unit_test/etl/algorithms/so/pso_variants/test_fs_pso.py`
  (etl-only, no torch) — 3 generations on 20-d Sphere via the shared
  `run_generations` driver; green.

## Verified ETL facts (do not re-investigate)
- **numpy arrays are NOT legal static trace inputs**: etl's `_is_static_value`
  accepts only None/bool/int/float/complex/str/Enum/dtype/slice/Dim/DimExpr/
  Device. Passing the FSPSO config (numpy lb/ub) to `etl.build`/`etl.run`
  raises TraceError. Workaround used here: register the config dataclass as a
  custom pytree node (`etl.register_pytree_node`) whose flatten turns numpy
  fields into Python-float leaves and unflatten rebuilds the arrays — the
  config then round-trips as static values and `run_generations` works
  unchanged. (Root-cause note: the DESIGN.md §4.3 "no partialization needed"
  claim only holds for scalar-leaf configs; the helpers' `run_generations`
  could alternatively closure-capture configs via `functools.partial`.)
- Newaxis indexing (`x[None, :]`) raises TraceError — use `enp.expand_dims`.
- `etl.min(x, axes=0)` on (n,) returns a scalar; `etl.argmin(x, axis=0)`
  returns an int64 scalar.
- `random.split_n(key, n)` returns n keys; `random.randint(key, (n,), low,
  high, dtype=etl.int32)`; `random.uniform/normal(key, shape, ..., etl.float32)`.
- `etl.select(cond, a, 0)` broadcasts python scalars; int32/int64 1-D indices
  work with `etl.gather(x, idx, axis=0)` (numpy-take semantics).
- `enp.zeros((n,), etl.float32)` and `enp.full((), float("inf"),
  dtype=etl.float32)` work inside traces.
- Draws are in torch order, one subkey each: rg, rp, tournament1, tournament2,
  offset, mutation_prob (then store the advanced key back in the state).
- torch `step` concatenates 2*half rows, so odd `pop_size` shrinks to
  2*(pop_size//2) after the first step — ported as-is (torch parity).

## Routing Table
| Area | Path |
|---|---|
| FS-PSO port | `fs_pso.py` |
| Shared helpers (min_by — temporary) | `utils.py` |
| Tests | `../../../../unit_test/etl/algorithms/so/pso_variants/` (sibling tree) |
| Torch reference (READ-ONLY) | `../../../../evox/algorithms/so/pso_variants/` |
| Operator/util shims | `../../_operator_shims.py` |
