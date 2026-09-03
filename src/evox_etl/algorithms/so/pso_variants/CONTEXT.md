# evox_etl/algorithms/so/pso_variants — functional PSO family

## Intent
Ports of the 7 torch PSO-family algorithms (read-only reference at
`../../../../evox/algorithms/so/pso_variants/`) to `init/ask/tell` plain
functions + frozen config/state dataclasses. See `../../../../DESIGN.md` §4-5.

## Status
- `utils.py` — DONE: `min_by(values, keys)`, `random_select_from_mask(key, mask,
  count, dim=-1)` (plain functions, key-first RNG).
- `pso.py` — DONE: `PSO` config + `PSOState` + `init/init_ask/init_tell/ask/
  tell`. Smoke test + torch parity test green.
- Remaining 6 variants (clpso, cso, dms_pso_el, fs_pso, sl_pso_gs, sl_pso_us)
  — pending (another task adds `__init__.py` after all 7 land).

## Notes for agents (verified against etl — do not re-investigate)
- **ndarray-in-config workaround (IMPORTANT)**: etl's tracer flattens config
  dataclasses at the build/run boundary and REJECTS numpy-array leaves
  (`ndarray` is not in etl's static-value whitelist: None/bool/int/float/
  complex/str/Enum/dtype/slice/Dim/DimExpr/Device). Configs holding `lb`/`ub`
  numpy arrays therefore cannot be passed to `etl.build`/`etl.run` as-is.
  `pso.py` works around it with `etl.core.register_pytree_node(PSO, _flatten,
  _unflatten)` (module-level, one registration per config TYPE): flatten
  surfaces the scalar fields + bound arrays as plain float leaves (static,
  accepted) and carries the original config in the context; unflatten returns
  the context object unchanged (configs are frozen, so identity round-trip is
  sound; context is never compared at the run boundary). Copy this pattern
  (`_register_config_pytree` in `pso.py`) into each new variant module for its
  own config type. Minimal repro of the underlying rejection: `etl.build(f,
  PSO(100, np.full(20, -10., np.float32), np.full(20, 10., np.float32)))` →
  `TraceError: ... (type ndarray) is neither a core.TensorSpec nor a static
  Python value`.
- Constants: bake bounds per call via `etl.ops.constant(etl.core.tensor(
  np.asarray(config.lb, dtype=np.float32)[None, :]))` — closure-captured
  concrete tensors are illegal at trace time.
- NO `x[None, :]`/newaxis indexing in traces — use `enp.expand_dims(x, axis=n)`
  (etl getitem supports only static ints/slices).
- `etl.gather(x, idx, axis)` is numpy-take semantics; `etl.scatter(x, idx, v,
  axis)` is replacement-only; `etl.select` requires a BOOL pred and NEP50-style
  promotes Python-scalar branches toward the tensor branch dtype.
- Draw order mirrors torch exactly (e.g. PSO: pop then velocity in init; rg then
  rp in step); split one subkey per draw. `random.uniform(key, shape, 0.0, 1.0,
  etl.float32)`.
- `min_by` (utils.py) is for 1-D keys (argmin over the concatenated axis 0).
- Parity caveat: etl Threefry keys vs torch MT19937 are different streams — with
  ~20 gens on Sphere dim=40 both sides sit in a high-variance regime (~100,
  paired ratios up to ~2x). At 100 gens both converge to single digits (seed 0:
  etl 4.15 vs torch 7.64), which is the regime the parity test uses.
