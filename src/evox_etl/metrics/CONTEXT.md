# evox_etl/metrics — multi-objective quality indicators

## Intent
Pure **plain** Python functions (NOT `@evoxc.defn` — DESIGN.md §4.3: defn objects
raise when nested, plain callables compose inside traces) implementing the
EvoX metric definitions on evoxc tensors. All functions must be called inside
an active trace (`evoxc.build`/`evoxc.run`); there is no eager mode.

## API Surface
- `gd(objs, pf)` — Generational Distance; L2 norm of per-solution min distance
  to the front, divided by n.
- `gd_plus(objs, pf, p=1.0)` — jax-evoX v0.9.0 GD+: distance per solution =
  `sqrt(sum(max(pf - obj, 0)^2))`; `(sum(min_dis**p)/n)**(1/p)`.
- `igd(objs, pf, p=1.0)` — IGD; mean over front points of min distance to the
  solutions, powered by p.
- `igd_plus(objs, pf, p=1.0)` — jax-evoX v0.9.0 IGD+: distance per front point
  = `sqrt(sum(max(obj - pf, 0)^2))` (reversed subtraction vs gd_plus);
  `(sum(min_dis**p)/k)**(1/p)`.
- `hv(key, objs, ref, num_sample=100000)` and
  `bounding_cube_monte_carlo_hv(key, objs, ref, num_sample=100000)` — the
  bounding-cube Monte Carlo estimator; `hv` delegates to
  `bounding_cube_monte_carlo_hv`. `key` is an evoxc PRNG key (first param —
  evoxc RNG needs keys). Both share the `_distances_to_ref`
  helper, which applies `clamp(ref - objs, min=0)`: objectives
  along which a point does not dominate the reference point contribute a 0
  extent, so such points do not add to the hypervolume.
- `each_cube_monte_carlo_hv(key, objs, ref, num_sample=100000)` — jax-evoX
  v0.9.0 per-cube Monte Carlo (no vmap): static Python loop over the n points
  (n is a static shape int), `random.split_n` keys, per-cube uniform draws,
  `sum(1/dom_count over dominating samples)` share counting.
- `_distance.py` — internal `pairwise_euclidean(x, y)`: `(n, m)` vs `(k, m)` →
  `(n, k)` via expand_dims broadcast (evoxc has no `cdist`).

All outputs cast to float32 explicitly (evoxc dtype promotion is stricter than
numpy — CONTEXT issue #12). `any`/`all` do not exist in evoxc: composed as
`evoxc.max`/`evoxc.min` over boolean axes.

## Constraints
- No torch/numpy imports. `import evoxc`, `import evoxc.numpy as enp`,
  `import evoxc.random as random` are the sanctioned imports.
- Broadcast tensor math only (no vmap). Static Python loops over static shape
  ints (num_points, m) are fine; no control flow on tensor values.
- Everything must run inside traces — validated via `evoxc.build` +
  `evoxc.run` (see unit_test conventions; `evoxc.run` returns concrete
  `Tensor` objects, use `.numpy()`).

## Validation (throwaway, results)
- gd/igd vs an independent float64 numpy implementation (cases (7,10,3),
  (1,5,2), (20,30,4), p=1,2): max |diff| 3.0e-8 / 2.4e-7.
- gd_plus/igd_plus vs numpy reference of the v0.9.0 formulas (p=1,2,4):
  exact (0.0).
- hv: bit-exact vs float32 recomputation from the returned sample matrix
  (float64 reference differs ~1.1e-6 — expected float32 rounding);
  0 <= hv <= max_vol.
- each_cube (n=3, num_sample=3000): exact vs numpy brute force.
- ref-clipping behaviour (upstream test cases + edges "point at ref",
  "duplicate outside point", "all points outside on one dim", 3-objective mixed
  front; 5 seeds, num_sample 100k-400k): Monte Carlo agreement within 6e-4
  (sampling noise), and all-zero cases exact (0.0).

## See Also
- `../DESIGN.md` §4.3 (plain functions), §6 (testing).
- `../../unit_test/etl/metrics/` — pytest suite (canonical mirror):
  `test_metrics.py` (numpy references only).
  Run: `python -m pytest unit_test/etl/metrics -q`.

## EvoX Compiler issues found
Consolidated in `../CONTEXT.md` "EvoX Compiler issues found" (#16-18:
`evoxc.run` returns concrete Tensors — use `.numpy()`; static build/run
arg-structure matching; positive finds re uniform/split_n/norm/gather/any-all).
