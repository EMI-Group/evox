# evox_etl/operators/mutation — pure functional mutation operators

## Intent
Mutation operators as PLAIN functions (NO `@evoxc.defn` — plain functions compose
inside traces; defn objects raise when called). API rules (binding,
`../../DESIGN.md` §4.3): keep the argument names/order EXACTLY; functions using
randomness gain `key` as FIRST parameter (`evoxc.random`; split from the caller's
key); the op mapping is where→select, pow→power, logical and→logical_and.

## Files
- `pm_mutation.py` — `polynomial_mutation(key, x, lb, ub, pro_m=1.0, dis_m=20.0)`:
  PlatEMO-style polynomial mutation. Two-phase select-chain (mu<=0.5 / mu>0.5),
  output clamped to [lb, ub].
- `__init__.py` — `__all__ = ["polynomial_mutation"]`, re-export.

## API Surface
- `polynomial_mutation(key, x, lb, ub, pro_m, dis_m) -> (n, d) float32 tensor`.
  `key`: evoxc int64 scalar key tensor (e.g. `evoxc.random.key(42)`).

## Gotchas (verified)
- No `evoxc.full/zeros/ones/empty` inside traces (concrete eager tensors raise in graph
  ops) — use select-chains with scalar `0.0` for the where-else branch.
- `evoxc.run(exe, key, x, lb, ub, pro_m, dis_m)` — static scalar args (pro_m, dis_m)
  must be re-passed to `evoxc.run` (they are part of the run signature), and the graph
  SPECIALIZES on their values at build time: re-passing a different scalar (e.g.
  pro_m=0.0 on a graph built with 1.0) raises `evoxc.core.TraceError` — build a
  separate executable per distinct static value.
- Key spec for `evoxc.build`: `evoxc.core.TensorSpec((), "int64")`.
  (`TensorSpec.from_tensor` does NOT exist in current evoxc.)
- `evoxc.random.key(seed)` returns an eager int64 scalar Tensor (creatable outside
  traces); `split`/`uniform` only work inside traces.
- numpy arrays are accepted as `evoxc.run` inputs on the numpy (cpu) backend;
  outputs convert via `out.numpy()` (np.asarray does NOT work).
