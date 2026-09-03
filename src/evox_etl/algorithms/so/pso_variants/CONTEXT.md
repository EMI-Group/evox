# evox_etl/algorithms/so/pso_variants — functional PSO variant algorithms

## Intent
Plain-function ports (`init`/`init_ask`/`init_tell`/`ask`/`tell` + frozen config/state
dataclasses, NO `@etl.defn`) of the torch PSO variants
(`../../../evox/algorithms/so/pso_variants/`, READ-ONLY reference). See
`../../../DESIGN.md` §4-5.

## API Surface
- `sl_pso_gs.py` — SLPSOGS (social learning PSO, Gaussian demonstrator choice):
  `SLPSOGS` config, `SLPSOGSState`, `init/init_ask/init_tell/ask/tell`.
- `sl_pso_us.py` — SLPSOUS (social learning PSO, uniform-sampling demonstrator
  choice): `SLPSOUS` config, `SLPSOUSState`, `init/init_ask/init_tell/ask/tell`.
- `utils.py` — TEMPORARY minimal `min_by` (concat axis 0 → `etl.argmin(keys,
  axis=0)` → gather with reshaped (1,) index → reshape back). The full
  `evox_etl.utils` package is being built in parallel; if a merge conflict on
  this file arises, the manager resolves it.

## Constraints / notes for agents (verified against etl — do not re-investigate)
- Configs store `lb`/`ub` as float TUPLES (converted from numpy arrays in
  `__post_init__` via `object.__setattr__`): the etl tracer rejects
  `np.ndarray`/numpy-scalar leaves in static trace args (`TraceError: ...
  is neither a core.TensorSpec nor a static Python value`). Constructor API is
  unchanged (numpy arrays in). This affects EVERY algorithm config with bounds
  — see "ETL issues found" in `../../../CONTEXT.md` for escalation.
- `etl.getitem` does NOT support `None` (newaxis) indices — use
  `enp.expand_dims` (e.g. `enp.expand_dims(state.global_best_fit, 0)`).
- Bounds baked per function via `etl.ops.constant(etl.core.tensor(
  np.asarray(config.lb, dtype=np.float32)[None, :]))`; numpy allowed ONLY for
  this constant baking.
- `etl.min` takes `axes=`; `etl.argmin`/`etl.argsort` take `axis=`. `etl.gather`
  is numpy-take semantics (1-D index → row selection along axis 0).
- torch `fit[argsort(fit, descending=True)]` == `etl.gather(pop,
  etl.argsort(-fit, axis=0, stable=True), axis=0)`; use `stable=True`.
- Key discipline: `key, subkey = random.split(state.key)`, one split per draw,
  store the advanced key back in the state; GS draw order = standard-normal
  (subkey1) then r1/r2/r3 (subkeys 2-4); US draw order = uniform_distribution
  (subkey1) then r1/r2/r3.
- `init_tell` updates ONLY `global_best_fit = etl.min(fitness, axes=0)` (torch
  `init_step` does not update the gb location — ported 1:1).
- Smoke tests: `unit_test/etl/algorithms/so/pso_variants/test_sl_pso_gs.py` +
  `test_sl_pso_us.py` (no torch; use `helpers.run_generations` +
  `SphereConfig` from `unit_test/etl/algorithms/helpers.py`; nested test files
  must add `REPO_ROOT/unit_test/etl/algorithms` to `sys.path` to import
  helpers).
