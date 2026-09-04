# evox_etl/operators — pure functional genetic operators

## Intent
Port of torch evox operators (`../../../evox/operators/`, read-only) as PURE functions
(plain Python, NOT `@etl.defn` — they get called inside traces by algorithms; a bare
call outside a trace raises etl's "no eager mode" TraceError, which is expected).
Porting rule (binding, see `../../DESIGN.md` §4.3):
- Keep torch function names, argument names and order EXACTLY.
- Functions using randomness in torch gain `key` as the FIRST parameter
  (etl.random; split from the caller's key).
- Drop the `device: torch.device` parameter (`latin_hypercube_sampling_standard`).
- `torch.Tensor` → etl tensors; translate ops 1:1 (where→select, clamp→clamp,
  argsort→argsort, gather→gather, cumsum→cumsum, rand→random.uniform etc.).

One exception to the "torch operators package" scope: `jit_fix_operator.py`
lives here too. It ports the torch `evox/utils/jit_fix_operator.py` helpers
(clamp/clamp_float/clamp_int, maximum/minimum(+`_int`), lexsort, nanmin/nanmax,
key-first randint, `_take_along_axis`) that torch keeps under `evox/utils/` —
the torch algorithms call these instead of raw torch ops for JIT-operator-fusion
safety, so the etl algorithm ports need them as well, and this package is their
canonical operator-utility home. Like everything else here it is plain Python
(trace-only) and is NOT re-exported from `__init__.__all__` — algorithms import
it directly (`from evox_etl.operators.jit_fix_operator import ...`).

## Status
ALL 15 torch operator functions are ported (git history from aa50f628 through
f9c5e612) and verified on the numpy backend: exact parity vs torch (1e-6) for the
deterministic functions, determinism/shape/bounds properties for the random ones
(30-check consolidated verification — both validation-gate scripts included).

## API Surface
Top-level `__init__.py` mirrors torch exactly:
`__all__ = ["crossover", "mutation", "sampling", "selection", "crowding_distance", "non_dominate_rank"]`.
- `sampling/`: `uniform_sampling(n, m) -> (weights, num_weights)` (Das-Dennis,
  deterministic), `grid_sampling(n, m)` (deterministic),
  `latin_hypercube_sampling_standard(key, n, d, smooth=True)` (key FIRST, no device
  arg), `latin_hypercube_sampling(key, n, lb, ub, smooth=True)`
- `selection/`: `dominate_relation(x, y)`, `non_dominate_rank(x)` (0-based ranks;
  iterative Pareto-front loop via `etl.while_loop` — torch's vmap/compile tricks
  replaced by the loop), `crowding_distance(costs, mask)`,
  `nd_environmental_selection(x, f, topk)`, `tournament_selection(key, n_round,
  fitness, tournament_size=2)`, `tournament_selection_multifit(key, n_round,
  fitnesses, tournament_size=2)`, `select_rand_pbest(key, percent, population,
  fitness)`, `apd_fn(x, y, z, obj, theta)`, `ref_vec_guided(x, f, v, theta)`.
  `dominate_relation`/`apd_fn` are NOT exported (mirrors torch).
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
| Jit-fix operator utils | `jit_fix_operator.py` | port of torch evox/utils/jit_fix_operator.py (clamp family, maximum/minimum, lexsort, nanmin/nanmax, randint, `_take_along_axis`); imported directly by algorithms, not in `__all__` |
| Tests (sibling) | `../unit_test/etl/operators/` | pending — root agent owns; see Test Strategy |

## Notes for Agents (cross-cutting etl gotchas)
Detailed per-module gotchas live in each subdir's CONTEXT.md. Cross-cutting:
- `etl.build(fn, *specs)` + `etl.run(exe, *args)`: positional args in order; tensor
  args declared via `etl.core.TensorSpec(shape, dtype)`; statics (int/float/None/
  bool) passed as-is and must be re-passed at run; graphs specialise on static
  values. Scalar tensor inputs must be 0-d ndarrays (np.float32(0.5) is REJECTED).
- `etl.gather` = numpy-take (out.shape = x.shape[:axis] + indices.shape +
  x.shape[axis+1:]); torch take_along_axis → `_take_along_axis` flatten trick
  (private helper in `selection/non_dominate.py`, imported by tournament/rvea).
- `etl.constant(etl.core.tensor(list, dtype=...))` is the only way to embed
  constants; `etl.full/zeros/ones/empty` are eager and raise inside traces — use
  select-chains with scalar `0.0` instead.
- dtype promotion differs from torch: `float32 * int32 → float64`, `int32 + python
  int → int64`. Cast explicitly to keep float32 outputs (see sbx.py sign cast).
- No `.ndim` on SymbolicTensor — check `getattr(x, "shape", None) is not None and
  len(x.shape) == k`. `enp.sum`/`enp.min` take `axis=` (singular). No
  newaxis/ellipsis in getitem (use `etl.reshape`/`enp.expand_dims`). `!=` on
  symbolic tensors → `etl.not_equal`. No where/any/all/amax/amin/meshgrid/atan2/
  moveaxis/repeat/take/lexsort; `log1p`/`floor` are at etl top level (not enp).
- `random.key(seed)` works eagerly; key spec is `TensorSpec((), "int64")`. Keys are
  never consumed — same key+inputs → identical output.
- `torch.randn`-based masks (DE_binary) → `random.normal`.

## Design Decisions
- `DE_differential_sum`: the torch file at HEAD is the OLD 4-arg form, but
  DESIGN.md §4.3 lists the newer `F`/`replace` form → port implements `key` +
  `F=None` + `replace=False`; `F=None` reproduces old torch behavior exactly.
- `polynomial_mutation`: the actual torch file uses `(x, lb, ub, pro_m, dis_m)`;
  DESIGN.md §4.3's "boundary" list entry is stale. Port follows the torch file
  (the binding rule).
- SBX/SBX-half offspring may legitimately lie OUTSIDE the parent range (beta > 1
  in the mu > 0.5 branch) — same as torch; do NOT test that invariant. Valid
  invariants: sbx symmetry `c1 + c2 == p1 + p2`; DE children elementwise ∈
  {mutation, current}; PM output within [lb, ub].
- `ref_vec_guided` can return NaN rows for empty reference-vector partitions —
  faithful torch parity; use `equal_nan=True` in comparisons.

## Test Strategy
Verified by consolidated scripts (30 checks, numpy backend): 9 deterministic
parity checks vs torch (1e-6) + 16 random-op property checks + both validation-gate
scripts (non_dominate_rank on (32,3), simulated_binary on (64,10) inside etl
graphs). The permanent unit-test suite goes under `../unit_test/etl/operators/`
(sibling — outside this node's write scope; full spec handed to the root agent):
parity tests importing torch live in `parity/` only; random-op tests need no torch.
Gate: `pytest unit_test/etl/operators -q` green.
