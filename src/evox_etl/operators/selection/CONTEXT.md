# evox_etl/operators/selection — pure functional selection operators

## Intent
Selection operators as PLAIN functions (no `@evoxc.defn` — defn objects raise
when called; plain functions compose inside traces). API rule: keep the function
names and argument order EXACTLY; functions using randomness gain `key` as FIRST
parameter (evoxc.random); everything else runs inside an active trace via
`evoxc.build(fn, specs...)` / `evoxc.run(exe, inputs...)`.

## Files (1.x file names kept; formulas verified)
- `non_dominate.py`: `dominate_relation`, `non_dominate_rank` (iterative
  Pareto-front loop via `evoxc.while_loop` — carries must be tensors, current
  rank carried as 0-d int32 `evoxc.constant`), `crowding_distance` (mask may be
  None → all-True constant), `nd_environmental_selection`.
- `tournament_selection.py`: `tournament_selection(key, n_round, fitness,
  tournament_size=2)`, `tournament_selection_multifit(key, n_round, fitnesses,
  tournament_size=2)` — key FIRST.
  Both return a 1-D `(n_round,)` int index tensor for every `n_round` (including 1:
  the single selected column is reshaped, never squeezed away), so downstream
  gather/broadcast sees the same shape.
- `find_pbest.py`: `select_rand_pbest(key, percent, population, fitness)`.
- `rvea_selection.py`: `apd_fn`, `ref_vec_guided` (no rng).
- `__init__.py`: the documented `__all__` (dominate_relation/apd_fn stay
  module-level, not exported).

## Private helpers (in non_dominate.py — imported by tournament_selection/rvea_selection)
- `_take_along_axis(x, indices, axis)` — gather-along-axis for 1D/2D x via
  flatten trick (`idx * stride + arange[None,:]` / `arange[:,None] * stride +
  idx` then `evoxc.gather` on `reshape(x, (-1,))`). Result shape = indices.shape.
- `_lexsort(keys, axis=0)` — the lexsort helper: stable argsort of
  keys[0], then per-key stable argsort of take_along_axis(key, indices)
  (numpy semantics: LAST key primary — matches np.lexsort).

## Dtype/shape notes (binding — verified against evoxc)
- `enp.sum`/`enp.min` take `axis=` (singular), NOT `axes=`.
- evoxc getitem does NOT support `None`/newaxis — use `evoxc.reshape` for
  `[:, None]` / `[None, :]` broadcasting.
- `!=` on symbolic tensors raises (delegates to `not __eq__` → `__bool__`) —
  use `evoxc.not_equal`.
- `evoxc.gather` = numpy-take (not a gather-along-axis); use `_take_along_axis` where
  a 2-D index is needed (crowding_distance sorted costs).
  numpy-take with 0-d index drops the axis (verified).
- `evoxc.scatter` = numpy `put_along_axis` semantics (indices padded to full
  rank); ops-level check expects updates rank = len(x.shape[:axis] +
  indices.shape + x.shape[axis+1:]) — for the crowding-distance scatter use a
  1-D init + flattened 1-D indices/updates.
- `int32 + python_int` promotes to int64 — cast increments explicitly
  (`evoxc.cast(cr + 1, 'int32')`).
- `evoxc.while_loop(cond_fn, body_fn, init)` passes the whole carried tree as a
  SINGLE argument; cond must return a 0-d bool scalar.
- Ranks are int32; argsort/argmin indices are int64; dominance/rank arithmetic
  uses int32 casts (`enp.sum(evoxc.cast(pred, 'int32'), axis=...) > 0` — evoxc has
  no any/all reductions).
