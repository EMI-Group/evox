# evox_etl/algorithms/so/pso_variants — PSO-family algorithms (functional)

## Intent
Plain-function ports of the torch pso_variants algorithms (read-only reference
in `../../../../evox/algorithms/so/pso_variants/`) to init/ask/tell functions +
frozen config/state dataclasses. See `../../DESIGN.md` §4-5 and the
`cso.py`/`clpso.py` modules for the established pattern.

## Files
- `cso.py` — CSO port (+ `CSO`, `CSOState`).
- `clpso.py` — CLPSO port (+ `CLPSO`, `CLPSOState`).
- `utils.py` — `min_by` (etl port of the torch pso_variants `utils.py`).
- Tests: `../../../../unit_test/etl/algorithms/so/pso_variants/` (sibling of
  `src/`, driven through `unit_test/etl/algorithms/helpers.py::run_generations`).

## Notes for agents (verified against etl — do not re-investigate)
- Configs holding numpy arrays (lb/ub/mean/stdev) CANNOT cross the etl trace
  boundary: numpy arrays are neither TensorSpecs nor static values, and
  dataclasses flatten to their fields. Register each numpy-holding config
  class as a zero-child pytree node:
  `etl.register_pytree_node(Cfg, lambda c: ((), c), lambda c, _children: c)`
  (see `cso.py`). The config then lives in the tree context and is baked as
  graph constants via `etl.ops.constant(etl.core.tensor(np.asarray(...)[None, :]))`;
  `etl.run` validates only its type, so re-passing it per run is harmless.
  State dataclasses must NOT be registered (they carry tensor leaves).
- etl `__getitem__` supports ints/slices but NOT `None`/newaxis/ellipsis
  ("None in the index key is not supported"). Use `enp.reshape(x, (n, 1))`
  for `[:, None]` and `enp.reshape(x, (1, d))` for `[None, :]`.
- etl has no `index_select`/`squeeze` — `min_by` uses gather with a reshaped
  (1,) argmin index and reshapes back.
- Key discipline: split the state key once per draw in torch draw order;
  store the last advanced key back into the state (JAX-evoX style).
- `etl.gather(x, idx, axis=0)` is numpy-take semantics (1-D idx == x[idx]);
  `etl.scatter(x, indices, updates, axis=0)` is replacement-only and accepts
  1-D int64 indices. `etl.min/mean/sum` take `axes=`, `etl.argmin` takes
  `axis=`.
