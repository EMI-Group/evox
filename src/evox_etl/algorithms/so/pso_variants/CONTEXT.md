# evox_etl/algorithms/so/pso_variants — functional PSO variants

## Intent
Ports of the torch evox PSO variants (read-only reference in
`../../../../evox/algorithms/so/pso_variants/`) to `init/ask/tell` plain functions +
frozen config/state dataclasses per `../../../DESIGN.md` §4-5.

## Status
- `dms_pso_el.py` — DONE (first algorithm port in evox_etl). Smoke test:
  `unit_test/etl/algorithms/so/pso_variants/test_dms_pso_el.py` (3 scenarios, green).
  Update math verified BIT-EXACT against the torch reference formulas for all three
  update paths (identical inputs + injected draws), including the quirks below.
- Remaining PSO variants (clpso, cso, dms_pso_el done, fs_pso, pso, sl_pso_gs,
  sl_pso_us, utils) — TODO, follow the dms_pso_el.py template.

## Notes for agents (verified against etl — do not re-investigate)
- **Config dataclasses CANNOT carry `np.ndarray` fields** through `etl.build`/
  `etl.run`: the tracer flattens user dataclasses as pytrees and rejects ndarray
  leaves (only TensorSpec or static Python scalars are legal trace inputs).
  Workaround used by DMSPSOEL: `__post_init__` normalizes `lb`/`ub` to
  `tuple(float, ...)` (accepting ndarray at construction); functions convert back
  with `np.asarray(config.lb, dtype=np.float32)`. Escalated to the root agent —
  see "ETL issues found" in `src/evox_etl/CONTEXT.md` (update there when writable).
- etl getitem supports ONLY static ints and slices — `x[:, None]` raises
  TraceError; use `enp.expand_dims(x, axis=1)`. Integer indexing mid-axis
  (`d[:, 0, :]`) and negative-free slices are fine.
- Data-dependent Python `if`s become `etl.cond(pred, true_fn, false_fn, *operands)`
  with `functools.partial` branches closing over the static config. Both branches
  are TRACED at compile time (draws inside them are graph ops on the symbolic key,
  executed at runtime only for the taken branch); branch outputs must be the same
  pytree with tensor leaves only. Frozen dataclasses work as cond operands/results
  (`dataclasses.replace` to update).
- RNG: split the key inside whichever branch executes; one `random.split` subkey
  per draw, torch draw order preserved, advanced key stored in the returned state
  (`key, key_a = random.split(state.key); key, key_b = random.split(key); ...`).
  `random.permutation(key, n, dtype=etl.int64)` takes a static int `n`.
- Reductions: `etl.sum/min/mean` take `axes=`; `etl.argmin/argmax` take `axis=`.
  `etl.argsort(x, axis=0, stable=True)` returns int64. `etl.remainder` promotes
  int32→int64 — use `etl.cast` when the leaf dtype must stay int32
  (e.g. iteration counter); python-float comparisons with int32 scalars
  (`iteration < 0.9 * max_iteration`) work on the numpy backend.
- `enp.full((), 0, dtype=etl.int32)` (0-d) works; `enp.zeros`/`enp.full` are
  symbolic (unlike etl.zeros/full which are concrete); `enp.zeros_like` does NOT
  exist.
- `etl.gather(x, idx, axis)` is numpy-take semantics (torch `x[idx]`).
- `init_tell` increments the iteration counter AFTER storing fitness (torch
  `init_step`); `ask` increments after the strategy switch (torch `step`).
- DMSPSOEL quirks ported 1:1: `_regroup` computes `regional_best_index` from the
  PRE-regroup dynamic-swarm fit (`state.fit[:dynamic_size]` — fit itself is never
  regrouped); the torch `sort_index[:dynamic_size]` assignment is dead code and
  omitted; strategy 2 leaves `local_best_*`/`regional_best_index` untouched.
- Import clamp from `evox_etl.algorithms._operator_shims`.
- Package has NO `__init__.py` (namespace package) — import as
  `evox_etl.algorithms.so.pso_variants.dms_pso_el`.

## Routing Table
| Area | Path |
|---|---|
| Torch reference (READ-ONLY) | `../../../../evox/algorithms/so/pso_variants/` |
| Tests | `../../../../unit_test/etl/algorithms/so/pso_variants/` |
| Shared operator shims | `../_operator_shims.py` |
