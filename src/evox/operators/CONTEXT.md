# evox/operators — pure functional genetic operators

## Intent
The genetic operators of the 2.0 package, as PURE functions
(plain Python, NOT `@evoxc.defn` — they get called inside traces by algorithms; a
bare call outside a trace raises evoxc's "no eager mode" TraceError, which is
expected).
API rule (binding, see `../../DESIGN.md` §4.3):
- Function names, argument names and order ARE the public API — keep them.
- Functions using randomness take `key` as the FIRST parameter
  (evoxc.random; split from the caller's key).
- There is no `device` parameter (`latin_hypercube_sampling_standard`).
- Everything operates on evoxc tensors (where→select, clamp→clamp, argsort→argsort,
  gather→gather, cumsum→cumsum, rand→random.uniform etc.).
Beyond the four operator families, `jit_fix_operator.py` lives here too: the
compatibility helpers (switch, clamp/clamp_float/clamp_int, clip,
maximum/minimum(+`_int`/`_float`), lexsort, nanmin/nanmax, key-first randint,
`_take_along_axis`) that algorithms call instead of raw tensor ops, which makes
this package their canonical operator-utility home. Like everything else here it
is plain Python (trace-only) and is NOT re-exported from `__init__.__all__` —
algorithms import it directly (`from evox.operators.jit_fix_operator import
...`).
## Status
All 15 operator functions are implemented and verified on the numpy backend:
1e-6 agreement with the documented formulas for the deterministic functions,
determinism/shape/bounds properties for the random ones.

## API Surface
Top-level `__init__.py` exports:
`__all__ = ["crossover", "mutation", "sampling", "selection", "crowding_distance", "non_dominate_rank"]`.
- `sampling/`: `uniform_sampling(n, m) -> (weights, num_weights)` (Das-Dennis,
  deterministic), `grid_sampling(n, m)` (deterministic),
  `latin_hypercube_sampling_standard(key, n, d, smooth=True)` (key FIRST, no device
  arg), `latin_hypercube_sampling(key, n, lb, ub, smooth=True)`
- `selection/`: `dominate_relation(x, y)`, `non_dominate_rank(x)` (0-based ranks;
  iterative Pareto-front loop via `evoxc.while_loop` — vmap/compile tricks
  replaced by the loop), `crowding_distance(costs, mask)`,
  `nd_environmental_selection(x, f, topk)`, `tournament_selection(key, n_round,
  fitness, tournament_size=2)`, `tournament_selection_multifit(key, n_round,
  fitnesses, tournament_size=2)`, `select_rand_pbest(key, percent, population,
  fitness)`, `apd_fn(x, y, z, obj, theta)`, `ref_vec_guided(x, f, v, theta)`.
  `dominate_relation`/`apd_fn` are NOT exported.
- `crossover/`: `DE_differential_sum(key, diff_padding_num, num_diff_vectors,
  index, population, F=None, replace=False)`, `DE_binary_crossover(key, ...)`,
  `DE_exponential_crossover(key, ...)`, `DE_arithmetic_recombination(mutation_vector,
  current_vector, K)` (deterministic, no key), `simulated_binary(key, x,
  pro_c=1.0, dis_c=20.0)`, `simulated_binary_half(key, x, pro_c=1.0, dis_c=20.0)`
- `mutation/`: `polynomial_mutation(key, x, lb, ub, pro_m=1.0, dis_m=20.0)`

## Routing Table
| Area | Path | Files |
|---|---|---|
| Sampling | `sampling/` | uniform.py (Das-Dennis), latin_hypercube.py, gird.py (grid) |
| Selection | `selection/` | non_dominate.py (dominate_relation, non_dominate_rank, crowding_distance, nd_environmental_selection), tournament_selection.py, find_pbest.py, rvea_selection.py |
| Crossover | `crossover/` | differential_evolution.py, sbx.py, sbx_half.py |
| Mutation | `mutation/` | pm_mutation.py |
| Jit-fix operator utils | `jit_fix_operator.py` | switch, clamp family, clip, maximum/minimum(+`_int`/`_float`), lexsort, nanmin/nanmax, randint, `_take_along_axis`; imported directly by algorithms, not in `__all__` |
| Tests (sibling) | `../unit_test/evox/operators/` | property tests (random-op properties + exact-formula checks) — root agent owns; see Test Strategy |

## Notes for Agents (cross-cutting evoxc gotchas)
Detailed per-module gotchas live in each subdir's CONTEXT.md. Cross-cutting:
- `evoxc.build(fn, *specs)` + `evoxc.run(exe, *args)`: positional args in order; tensor
  args declared via `evoxc.core.TensorSpec(shape, dtype)`; statics (int/float/None/
  bool) passed as-is and must be re-passed at run; graphs specialise on static
  values. Scalar tensor inputs must be 0-d ndarrays (np.float32(0.5) is REJECTED).
- `evoxc.gather` = numpy-take (out.shape = x.shape[:axis] + indices.shape +
  x.shape[axis+1:]); take_along_axis → `_take_along_axis` flatten trick
  (private helper in `selection/non_dominate.py`, imported by tournament/rvea).
- `evoxc.constant(evoxc.core.tensor(list, dtype=...))` is the only way to embed
  constants; `evoxc.full/zeros/ones/empty` are eager and raise inside traces — use
  select-chains with scalar `0.0` instead.
- dtype promotion is stricter than numpy: `float32 * int32 → float64`, `int32 + python
  int → int64`. Cast explicitly to keep float32 outputs (see sbx.py sign cast).
- No `.ndim` on SymbolicTensor — check `getattr(x, "shape", None) is not None and
  len(x.shape) == k`. `enp.sum`/`enp.min` take `axis=` (singular). No
  newaxis/ellipsis in getitem (use `evoxc.reshape`/`enp.expand_dims`). `!=` on
  symbolic tensors → `evoxc.not_equal`. No where/any/all/amax/amin/meshgrid/atan2/
  moveaxis/repeat/take/lexsort; `log1p`/`floor` are at evoxc top level (not enp).
- `random.key(seed)` works eagerly; key spec is `TensorSpec((), "int64")`. Keys are
  never consumed — same key+inputs → identical output.
- normal-draw masks (DE_binary) → `random.normal`.

## Design Decisions
- `DE_differential_sum` takes `key` first plus `F=None` + `replace=False`;
  `F=None` selects the default scaling factor.
- `polynomial_mutation`: canonical signature
  `polynomial_mutation(key, x, lb, ub, pro_m, dis_m)` (binding rule).
- SBX/SBX-half offspring may legitimately lie OUTSIDE the parent range (beta > 1
  in the mu > 0.5 branch) — expected; do NOT test that invariant. Valid
  invariants: sbx symmetry `c1 + c2 == p1 + p2`; DE children elementwise ∈
  {mutation, current}; PM output within [lb, ub].
- `ref_vec_guided` can return NaN rows for empty reference-vector partitions —
  expected NaN semantics; use `equal_nan=True` in comparisons.

## Test Strategy
The permanent suite is the sibling `../unit_test/evox/operators/` (numpy
backend): `test_crossover.py`, `test_mutation.py`, `test_sampling.py`,
`test_selection.py`, `test_jit_fix_operator.py` — random-op property tests plus
exact-formula checks (1e-6), with no torch imports anywhere.
Gate: `pytest unit_test/evox/operators -q` green.
