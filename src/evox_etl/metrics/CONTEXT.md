# evox_etl/metrics — multi-objective quality indicators

## Intent
Pure **plain** Python functions (NOT `@etl.defn` — DESIGN.md §4.3: defn objects
raise when nested, plain callables compose inside traces) implementing the
torch `evox.metrics` math on etl tensors. All functions must be called inside
an active trace (`etl.build`/`etl.run`); there is no eager mode.

## API Surface
- `gd(objs, pf)` — Generational Distance; L2 norm of per-solution min distance
  to the front, divided by n (torch `gd` semantics).
- `gd_plus(objs, pf, p=1.0)` — jax-evoX v0.9.0 GD+: distance per solution =
  `sqrt(sum(max(pf - obj, 0)^2))`; `(sum(min_dis**p)/n)**(1/p)`.
- `igd(objs, pf, p=1.0)` — IGD; mean over front points of min distance to the
  solutions, powered by p (torch `igd` semantics).
- `igd_plus(objs, pf, p=1.0)` — jax-evoX v0.9.0 IGD+: distance per front point
  = `sqrt(sum(max(obj - pf, 0)^2))` (reversed subtraction vs gd_plus);
  `(sum(min_dis**p)/k)**(1/p)`.
- `hv(key, objs, ref, num_sample=100000)` and
  `bounding_cube_monte_carlo_hv(key, objs, ref, num_sample=100000)` — torch
  `hv` bounding-cube Monte Carlo; `hv` delegates to
  `bounding_cube_monte_carlo_hv`. `key` is an etl PRNG key (first param —
  etl RNG needs keys, unlike torch).
- `each_cube_monte_carlo_hv(key, objs, ref, num_sample=100000)` — jax-evoX
  v0.9.0 per-cube Monte Carlo (no vmap): static Python loop over the n points
  (n is a static shape int), `random.split_n` keys, per-cube uniform draws,
  `sum(1/dom_count over dominating samples)` share counting.
- `_distance.py` — internal `pairwise_euclidean(x, y)`: `(n, m)` vs `(k, m)` →
  `(n, k)` via expand_dims broadcast (etl has no `cdist`).

All outputs cast to float32 explicitly (etl dtype promotion differs from
torch — CONTEXT issue #12). `any`/`all` do not exist in etl: composed as
`etl.max`/`etl.min` over boolean axes.

## Constraints
- No torch/numpy imports. `import etl`, `import etl.numpy as enp`,
  `import etl.random as random` are the sanctioned imports.
- Broadcast tensor math only (no vmap). Static Python loops over static shape
  ints (num_points, m) are fine; no control flow on tensor values.
- Everything must run inside traces — validated via `etl.build` +
  `etl.run` (see unit_test conventions; `etl.run` returns concrete
  `Tensor` objects, use `.numpy()`).

## Validation (throwaway, results)
- gd/igd vs torch evox (cases (7,10,3), (1,5,2), (20,30,4), p=1,2):
  max |diff| 3.0e-8 / 2.4e-7.
- gd_plus/igd_plus vs numpy reference of the v0.9.0 formulas (p=1,2,4):
  exact (0.0).
- hv: bit-exact vs float32 recomputation from the returned sample matrix
  (float64 reference differs ~1.1e-6 — expected float32 rounding);
  0 <= hv <= max_vol.
- each_cube (n=3, num_sample=3000): exact vs numpy brute force.

## See Also
- `../../evox/metrics/` — READ-ONLY torch reference.
- `../DESIGN.md` §4.3 (plain functions), §6 (testing).
- `tests/` — pytest suite (relocation-ready conftest, see
  `problems/tests/` precedent): `test_metrics.py` (numpy references, no torch)
  + `parity/test_parity.py` (torch parity; the only torch-importing file).
  Run: `python -m pytest src/evox_etl/metrics/tests -q` (53 tests).

## ETL issues found (to be merged into ../CONTEXT.md by the parent)
1. `etl.run` returns concrete `etl.core.tensor.Tensor` objects, NOT ndarrays —
   `np.asarray(result)` yields an object-dtype 0-d array; use `result.numpy()`.
2. Static positional args must be passed to `etl.build` AND re-passed (by
   value) to `etl.run`; parameters omitted at build (Python defaults) are
   baked into the executable and must NOT be passed at run (else "run-time
   input structure does not match the traced signature").
3. Positive finds (no workarounds needed): `etl.random.uniform` DOES accept
   tensor `low`/`high` bounds (broadcast); `etl.random.split_n(key, n)`
   returns a TUPLE of n keys (not a tensor) — ideal for static per-cube
   loops; `etl.norm(x)` with default `axis=None, ord=2` = L2 norm of a
   vector; float32 tensor `**` Python-float exponent stays float32 (no
   issue-#12 promotion); boolean reductions compose `any` → `etl.max(…,
   axes=…)`, `all` → `etl.min(…, axes=…)` (etl has no any/all ops).
