# evox_etl/utils — functional helpers

## Intent
Pure functions used across algorithms: tree helpers, `min_by`, `dominate_relation`,
`pairwise_euclidean_dist`/`pairwise_manhattan_dist`/`pairwise_chebyshev_dist`/
`cos_dist`, `rank`, `rank_based_fitness`, `cal_max`, `parse_opt_direction`,
`compose`. Mirror `../../evox/utils/` (read-only sibling) / JAX evox v0.9.0
`utils/common.py`. See `../DESIGN.md`.

## API Surface (`common.py`, re-exported by `__init__.py`)
ALL functions are PLAIN Python functions (NOT `@etl.defn`) — callable only
inside an active etl trace; a bare call outside a trace raises
`etl.core.TraceError` (expected). Exceptions: `parse_opt_direction` and
`compose` are pure Python and also work host-side.

- `min_by(values, keys)` — lists are concatenated along axis 0; returns
  (value, key) with the concatenation axis squeezed (scalar `etl.argmin` index
  + `etl.gather` numpy-take semantics; NO `values[idx]` dynamic scalar
  indexing — not supported in etl).
- `pairwise_euclidean_dist` / `pairwise_manhattan_dist` /
  `pairwise_chebyshev_dist` `(x, y)` — (n,d)×(k,d) → (n,k) via
  `enp.expand_dims(x, 1)` vs `enp.expand_dims(y, 0)` broadcasting + reduce
  over the LAST axis (the JAX vmap-in-vmap becomes direct broadcasting).
- `euclidean_dist` / `manhattan_dist` / `chebyshev_dist` / `pair_max` /
  `pairwise_func(x, y, func)` — per-pair primitives (reduce over last axis).
- `cos_dist(x, y)` — (n,d)×(k,d) → (n,k) row-normalized dot products per the
  v0.9.0 formula; zero-norm rows propagate nan like jnp (no clamping).
- `cal_max(x, y)` — (n,d)×(k,d) → (n,k) pairwise max of differences.
- `rank(array)` — 1-d int ranks: `etl.argsort` (int64) + `etl.scatter` with
  `order * 0` zero-init (no `*_like` ops) and `enp.arange(n,
  dtype=order.dtype)` updates.
- `rank_based_fitness(raw_fitness)` — float32 in [-0.5, 0.5]; explicit
  `etl.cast(rank, etl.float32)` BEFORE dividing (else int64/int → float64).
- `parse_opt_direction(opt_direction)` — "min"→1, "max"→-1 (ValueError
  otherwise); iterable → tuple of ±1 ints. Pure Python, usable in-graph on
  static config values.
- `compose(*functions)` — left-to-right composition; single-iterable unwrap.
- Tree re-exports: `tree_map`, `tree_leaves`, `tree_flatten`,
  `tree_unflatten` from `etl` top level (etl names; NOTE `tree_unflatten`
  arg order is (leaves, treespec) — REVERSED vs JAX).

`dominate_relation` is deliberately NOT here — operators/selection owns it.

## Signature notes (etl quirks to respect)
- `etl.norm` takes the SINGULAR `axis=` kwarg (reductions like sum/max take
  `axes=`, argmin/argmax/argsort take `axis=` — check each op). `norm` has
  `keepdims=True` — used by `cos_dist` instead of expand_dims.
- `etl.gather` has numpy-take semantics and ACCEPTS 0-d scalar indices
  (squeezes the axis) — verified on the shared venv etl.
- `etl.argsort` returns int64 indices; keep `rank` outputs int64 via
  `enp.arange(n, dtype=order.dtype)`.
- `etl.transpose` axes must be a tuple: `axes=(1, 0)`.
- No torch/numpy in graph code; numpy allowed only for baking constants.
