# evox_etl/operators/selection — pure functional selection operators

## Intent
Port of torch evox selection operators (`../../../evox/operators/selection/`,
read-only) as PLAIN functions (no `@etl.defn` — defn objects raise when called;
plain functions compose inside traces). Porting rule: keep torch function names
and argument order EXACTLY; functions using randomness gain `key` as FIRST
parameter (etl.random); everything else runs inside an active trace via
`etl.build(fn, specs...)` / `etl.run(exe, inputs...)`.

## Files (mirror torch names; parity vs torch verified in smoke tests)
- `non_dominate.py`: `dominate_relation`, `non_dominate_rank` (iterative
  Pareto-front loop via `etl.while_loop` — carries must be tensors, current
  rank carried as 0-d int32 `etl.constant`), `crowding_distance` (mask may be
  None → all-True constant), `nd_environmental_selection`.
- `tournament_selection.py`: `tournament_selection(key, n_round, fitness,
  tournament_size=2)`, `tournament_selection_multifit(key, n_round, fitnesses,
  tournament_size=2)` — key FIRST.
- `find_pbest.py`: `select_rand_pbest(key, percent, population, fitness)`.
- `rvea_selection.py`: `apd_fn`, `ref_vec_guided` (no rng).
- `__init__.py`: mirrors torch `__all__` exactly (dominate_relation/apd_fn stay
  module-level, not exported).

## Private helpers (in non_dominate.py — imported by tournament_selection/rvea_selection)
- `_take_along_axis(x, indices, axis)` — torch take_along_dim for 1D/2D x via
  flatten trick (`idx * stride + arange[None,:]` / `arange[:,None] * stride +
  idx` then `etl.gather` on `reshape(x, (-1,))`). Result shape = indices.shape.
- `_lexsort(keys, axis=0)` — port of torch evox lexsort: stable argsort of
  keys[0], then per-key stable argsort of take_along_axis(key, indices)
  (numpy semantics: LAST key primary — matches torch behavior and np.lexsort).

## Dtype/shape notes (binding — verified against etl)
- `enp.sum`/`enp.min` take `axis=` (singular), NOT `axes=`.
- etl getitem does NOT support `None`/newaxis — use `etl.reshape` for
  `[:, None]` / `[None, :]` broadcasting.
- `!=` on symbolic tensors raises (delegates to `not __eq__` → `__bool__`) —
  use `etl.not_equal`.
- `etl.gather` = numpy-take (NOT torch gather); use `_take_along_axis` where
  torch used torch.gather with 2D index (crowding_distance sorted costs).
  numpy-take with 0-d index drops the axis (verified).
- `etl.scatter` = numpy `put_along_axis` semantics (indices padded to full
  rank); ops-level check expects updates rank = len(x.shape[:axis] +
  indices.shape + x.shape[axis+1:]) — for the crowding-distance scatter use a
  1-D init + flattened 1-D indices/updates.
- `int32 + python_int` promotes to int64 — cast increments explicitly
  (`etl.cast(cr + 1, 'int32')`).
- `etl.while_loop(cond_fn, body_fn, init)` passes the whole carried tree as a
  SINGLE argument; cond must return a 0-d bool scalar.
- Ranks are int32; argsort/argmin indices are int64; dominance/rank arithmetic
  uses int32 casts (`enp.sum(etl.cast(pred, 'int32'), axis=...) > 0` — etl has
  no any/all reductions).
