# evox_etl/algorithms/mo — functional ports of the torch MO algorithms (step protocol)

## Intent
Plain-function ports of `src/evox/algorithms/mo/` (torch, READ-ONLY reference):
nsga2, nsga3, moead, rvea, rveaa, hype. Each module = frozen config dataclass
(`<Name>Config`), frozen state dataclass (`<Name>State`, tensor leaves only),
and plain `init`/`init_step`/`step` functions following the step protocol of
`../../core/algorithm.py` (NO `@etl.defn`, see `../../DESIGN.md` §4.3; no
module in this family defines `final_step`). Bindings: `../../DESIGN.md` §4-5.

## API Surface (all six modules)
- `init(config, key) -> state` draws the initial population and derived
  tensors (reference vectors, ideal point buffers, generation counter).
- `init_step(config, state, evaluate) -> state` — generation 0 evaluates the
  FULL initial population (verified against all six torch classes, HypE
  included, whose init_step also derives `ref = 1.2 * max(fitness)`); moead's
  additionally seeds the ideal point `z = min(fitness, axes=0)`; nsga2's runs
  the environmental selection over the initial population.
- `step(config, state, evaluate) -> state` owns ONE full generation, fused
  into a single trace:
  1. offspring phase (torch's candidate generation): mating pool →
     crossover → mutation → clamp; the offspring batch and the advanced
     key/generation are written into an INTERMEDIATE state via
     `replace(...)` (field names `offspring` / `off` / `next_generation`)
     because the selection phase needs the candidates but `evaluate` only
     returns fitness;
  2. `fitness = evaluate(offspring)` through the workflow-owned opaque
     closure (solution_transform → problem → opt-direction scaling →
     fitness_transform → monitor update; minimization semantics; never
     stored or re-threaded);
  3. selection phase: merge parents + offspring and run the environmental
     selection (split into a module-private helper in nsga3/moead — a plain
     function split, not a protocol function).
- The workflow dispatches `init_step`/`step` HOST-SIDE (no in-graph
  generation branch); every module traces via `etl.build`/`etl.run`.
- Config fields mirror torch `__init__` exactly minus `device`; `lb`/`ub`
  tuples baked as (dim,) float32 graph constants
  (`bake_bounds` → `etl.ops.constant(etl.core.tensor(np.asarray(...,
  np.float32)))`); optional ops are plain function refs, `None` = algorithm
  default, resolved inside functions (see the config-construction section
  for the registration/op-field split between the six modules).
- Operators are imported from the canonical torch-parity-verified modules:
  selection (`nd_environmental_selection`, `non_dominate_rank`,
  `tournament_selection[_multifit]`, `ref_vec_guided`) from
  `evox_etl.operators.selection`; `simulated_binary[_half]` from
  `evox_etl.operators.crossover`; `polynomial_mutation` from
  `evox_etl.operators.mutation` (canonical signature `(key, x, lb, ub,
  pro_m=1.0, dis_m=20.0)` — lb/ub passed directly, no boundary stack);
  `uniform_sampling` from `evox_etl.operators.sampling`; and `clamp`,
  `minimum`, `lexsort`, `nanmax`, `nanmin`, `randint`, `_take_along_axis`
  from `evox_etl.operators.jit_fix_operator` (canonical port of the torch
  `utils/jit_fix_operator` helpers).
- RNG: `key, subkey = random.split(state.key)` (returns TWO keys); several draws
  → `random.split_n(key, n)`; advanced `key` stored back. `init_step`/`step`
  are deterministic given the state (all randomness is drawn before the
  `evaluate` call).

## Config construction (current state — post make_* refactor, T2-A44)
- Public construction goes through module-level `make_*` constructors
  (`make_nsga2`, `make_nsga3`, `make_moead`, `make_rvea`, `make_rveaa`,
  `make_hype` — same module as each config; exported from `mo/__init__.py`
  alongside the `*Config` names). Configs are DUMB frozen dataclasses: NO
  `__post_init__` anywhere in mo/*.py, tuple-stored bounds, working
  `__eq__`/`__hash__` (RVEAaConfig was `eq=False` in the ndarray era, now
  plain `@dataclass(frozen=True)`; RVEAaState keeps `eq=False` — tensor
  fields).
- Bounds: `lb`/`ub` are stored as flat tuples of plain Python floats.
  Constructors accept any 1-D array-like (`ArrayLike`, `normalize_bounds`,
  `bake_bounds`, `bake_float32_constant` live in
  `evox_etl/algorithms/_config_utils.py`, shared across the SO/MO rounds)
  and raise ValueError on malformed input ("lb and ub must be 1-D...", "lb
  and ub must have the same shape..."). dtype is moot after storage: module
  bakers re-cast to float32 at trace. The old rveaa dtype-equality
  `__post_init__` assert is deliberately gone.
- Bounds baking: module-local `_bounds` defs are deleted; every module bakes
  via `lb, ub = bake_bounds(config.lb, config.ub)` (natural (dim,) float32
  graph constants); `dim = len(config.lb)`.
- Op fields: ALL SIX configs carry `Optional[Callable]` op fields mirroring
  torch's `__init__` (nsga2/nsga3/rvea/rveaa all three; hype `mutation_op` +
  `crossover_op` only — torch accepts `selection_op` there but immediately
  overwrites it with `tournament_selection`, so it is inert and intentionally
  not exposed; moead `mutation_op`/`crossover_op`, `selection_op`
  accepted-and-ignored because torch ignores it too). nsga3 additionally
  carries `data_type: Optional[Any] = None`. Every `step` resolves a `None` op
  field to the torch default at the torch call site; an all-None config is
  byte-identical to the pre-injection behavior (same RNG draw order).
  Custom-op signatures (etl keyed unless noted): `selection_op(key, n_parents,
  fitness)` for the nsga2/nsga3/moead tournament style, or `selection_op(x, f,
  v, theta)` with NO key for the rvea/rveaa ref-vector style;
  `crossover_op(key, x)`; `mutation_op(key, x, lb, ub)`.
- Callable-bearing configs KEEP a zero-child `etl.register_pytree_node`
  registration (all six modules now) — REQUIRED because non-None callables are
  NOT static pytree values (TraceError at the callable leaf). Registration
  makes the config an opaque static node: `etl.run` performs NO by-value
  revalidation, and `lb`/`ub`/op fields travel with the node. The
  `_config_flatten`/`_config_unflatten` boilerplate is duplicated per module by
  design (same convention as the operators).
- `make_nsga3` also raises ValueError when `data_type` is neither `None` nor
  the builtin `bool` type (`_normalize_data_type`, re-called in `init` so
  direct dataclass construction fails loudly too). `data_type=bool` draws the
  torch `data_type == torch.bool` boolean population (uniform > 0.5);
  `torch.bool` itself is NOT accepted (the etl module never imports torch) —
  callers must pass the builtin `bool`.
- `make_*` signatures (drives the test-migration round; all raise ValueError
  on malformed bounds; the four callable-bearing ones (nsga3/moead/rvea/rveaa)
  also raise ValueError on a non-callable non-None op field):
  - `make_nsga2(pop_size: int, n_objs: int, lb: ArrayLike, ub: ArrayLike) -> NSGA2Config`
  - `make_nsga3(pop_size: int, n_objs: int, lb: ArrayLike, ub: ArrayLike, selection_op: Optional[Callable] = None, mutation_op: Optional[Callable] = None, crossover_op: Optional[Callable] = None, data_type: Optional[Any] = None) -> NSGA3Config`
  - `make_moead(pop_size: int, n_objs: int, lb: ArrayLike, ub: ArrayLike, selection_op: Optional[Callable] = None, mutation_op: Optional[Callable] = None, crossover_op: Optional[Callable] = None) -> MOEADConfig`
  - `make_rvea(pop_size: int, n_objs: int, lb: ArrayLike, ub: ArrayLike, alpha: float = 2.0, fr: float = 0.1, max_gen: int = 100, selection_op: Optional[Callable] = None, mutation_op: Optional[Callable] = None, crossover_op: Optional[Callable] = None) -> RVEAConfig`
  - `make_rveaa(pop_size: int, n_objs: int, lb: ArrayLike, ub: ArrayLike, alpha: float = 2.0, fr: float = 0.1, max_gen: int = 100, selection_op: Optional[Callable] = None, mutation_op: Optional[Callable] = None, crossover_op: Optional[Callable] = None) -> RVEAaConfig`
  - `make_hype(pop_size: int, n_objs: int, lb: ArrayLike, ub: ArrayLike, n_sample: int = 10000) -> HypEConfig`
- Direct `*Config(...)` dataclass construction with ndarray lb/ub still runs
  (field annotations are unenforced and the installed etl accepts ndarray
  statics) but is not the sanctioned API — unit-test files are being
  migrated to `make_*` by a parallel test-migration round.

## Per-module state shapes and quirks
- Effective pop_size: MOEAD/RVEA/RVEAa overwrite torch `self.pop_size` with the
  Das-Dennis count `n_v` from `uniform_sampling(pop_size, n_objs)` (returns
  `(points, n_samples)` — the static int drives shapes). NSGA2/NSGA3/HypE keep
  the user pop_size. SBX quirk preserved: offspring batch = 2*(n//2) rows
  (torch `x[n//2 : n//2*2]`), e.g. 14 for n_v=15.
- RVEA: `pop` stays (n_v, dim) forever. RVEAa: after the first generation
  `pop` GROWS to (2*n_v, dim) — `ref_vec_guided` returns one row per
  reference vector (2*n_v after RV regeneration) and unmatched vectors
  produce NaN rows — until the final `gen == max_gen` generation, where the
  batch truncation keeps only the second half (rows ≥ n_v) and the remaining
  rows become NaN. Both: the mating pool always draws the FIXED n_v, and the
  torch `_mating_pool` valid-prefix trick spans `pop.shape[0]` rows — never
  derive n_v from `pop.shape[0]`.
- HypE keeps the torch algorithm split as module functions:
  `cal_hv(key, fit, ref, pop_size, n_sample)` (Monte-Carlo hypervolume
  contribution; `pop_size` is a Python int in the first call and a scalar
  tensor in the merged call) and the merged-batch truncation by
  `lexsort([-dis, rank])[:pop_size]`.
- NSGA3's environmental selection ends in module-level
  `_final_survivors(merge_pop, merge_fit, rank, worst_rank, pop_size)`: it
  keeps the `pop_size` rows with `rank < worst_rank` ordered by POSITION in
  the shuffled merge arrays, matching torch's mask selection
  `merge_pop[rank < worst_rank]` (an ascending-rank sort would give the same
  SET in a different order).

## Known Issues
- **NSGA3 with odd `pop_size` raises ShapeError at trace time** (`gather:
  index 2*pop_size is out of bounds for axis 0 with size 2*pop_size`):
  `simulated_binary` pairs `n//2` parents, so an odd offspring batch is
  2*pop_size−1 rows, while the merge/selection logic assumes exactly
  2*pop_size (the Das-Dennis `ref` also has n_v ≠ pop_size rows when
  `pop_size` is not a Das-Dennis count). PRE-EXISTING — reproduces
  identically in the pre-conversion code and matches the torch
  reference's own assumption; use even pop_size values.
- **RVEAa NaN survivor rows**: after the first generation `pop`/`fit` contain
  all-NaN rows (one per unmatched reference vector). These are torch
  semantics (`ref_vec_guided` writes NaN for null niches), not a port bug:
  the mating pool sorts NaN rows to the back with an int32-max sentinel, and
  `nanmin`/`nanmax` skip them. Consumers must tolerate NaN rows (the
  EvalMonitor is one such consumer that does not — see the escalation note).
- **HypE's `cal_hv`/`init_step`/`step` have un-annotated parameters/returns**
  (`def init_step(config: HypEConfig, state: HypEState, evaluate):` — no
  return annotation, `evaluate` and parts of `cal_hv` untyped), unlike the
  other five modules. Cosmetic only; left as-is to keep the diff minimal.
- **MOEAD's ideal point `z` makes it metric-hostile**: `z` is a loop carry
  updated per-i BEFORE the PBI comparisons (torch order — do NOT pre-lower z
  with the batch min, that contaminates the comparisons), and the returned
  `z` reflects only the minima seen inside the LAST `step` call's sequential
  update, not a global running min of all evaluated fitness. Metrics that
  need the true ideal point should compute it from the monitor history
  (min over `fit`) rather than reading `state.z`.
- **NSGA3's linalg is exporter-constrained**: the stablehlo-v1 exporter
  DEFERS `matrix_rank`/`svd`/`solve` → BackendError (escalated to the root
  agent). The torch `matrix_rank(extreme) == n_objs` guard is therefore
  rank = count(sqrt(eigvalsh(AᵀA)) > s_max·n·eps32) with AᵀA accumulated in
  float64 (noise floor ~1e-8·s_max « cutoff ~1e-7·s_max; verified ≡
  `etl.matrix_rank` on 14k random/singular/near-singular matrices), and the
  torch `solve(extreme, ones)` hyperplane is the eigh-based normal equations
  (AᵀA)⁻¹Aᵀ·1 in float64 (≤1.2e-7 rel vs the numpy LU solve even at cond
  1e5). Both build+run on iree-llvm-cpu and xla-cuda.

## ETL gotchas (verified — do not re-investigate)
- MOEAD's update loop is one traced `etl.while_loop`: cond/body take the
  carry as ONE tuple arg; closure-capture of outer args (`state.w`,
  `state.next_parents`, `fitness`) is legal. Carry `i` must stay int32
  (`etl.cast(i + 1, etl.int32)`).
- MOEAD sequential overwrites: the same row can be updated by several i's in
  one generation (last-i-wins) — torch does the same; do not "deduplicate".
- `etl.sum/mean` take `axes=`; `etl.min(x, axes=..)` values only (argmin
  separate); `etl.topk(x, k, axis, largest=False)` → `(values, indices)`;
  `etl.sort(x, axis, descending=, stable=)` values only; `etl.argsort(...,
  stable=True)` → int64 indices. `random.permutation(key, n, dtype=int32)`.
- `etl.cond(pred, true_fn, false_fn, *operands)` — NSGA3 uses it for the
  hyperplane branch: the full-rank guard keeps the solve off deficient
  `extreme` matrices (numpy `linalg.solve` would RAISE there; never compute
  it unconditionally).
- No boolean/advanced indexing with dynamic masks (no dynamic shapes): select
  survivors via sort/argsort of masked values + static `[:pop_size]` slice, or
  one-hot `reduce_max(equal(...))` masks for torch `_masked_assign` patterns.
- `etl.gather` = numpy `take`: row-local gathers → `_take_along_axis` from
  `evox_etl.operators.jit_fix_operator`; `etl.scatter` = put_along_axis
  replacement (no scatter-add — torch NSGA3 `scatter_add` → one-hot sum).
- No `&`/`!=`/`%` overloads → `enp.logical_and`, `etl.not_equal`,
  `etl.remainder`. `enp.zeros/enp.full` (not `etl.zeros`); Python-int tensor
  promotion → wrap in `etl.cast`. `etl.stack` of mixed dtypes is risky → cast
  int32 rank to float32 before stacking in `tournament_selection_multifit`.
- while_loop carry is ONE pytree argument (unpack inside cond/body); carry
  dtypes must be identical across iterations.
- No `etl.cdist` → MOEAD neighbors via
  `sqrt(maximum(d2, 0))`, `d2 = w2[:, None] + w2[None, :] - 2*dot(w, w.T)`.

## Tests
- Smoke: `unit_test/etl/algorithms/mo/test_<name>.py` (etl-only, `__init__.py`
  present so pytest collects uniquely-qualified names vs the parity dir).
- Parity (torch allowed): `unit_test/etl/algorithms/parity/test_nsga2.py`,
  `test_nsga3.py`, `test_moead.py` + shared `parity_common.py` (also a package).
- Gate: `/mnt/local-ssd/bchuang/evox/.venv/bin/python -m pytest
  unit_test/etl/algorithms/mo unit_test/etl/algorithms/parity/test_nsga2.py
  unit_test/etl/algorithms/parity/test_nsga3.py
  unit_test/etl/algorithms/parity/test_moead.py -q` — green.
- Test files construct configs directly today (ndarray bounds still work at
  runtime); a parallel test-migration round owns switching them to `make_*`.

## Routing Table
| Area | Path |
|---|---|
| NSGA2 | `nsga2.py` |
| NSGA3 (tensorized, reference-point niching) | `nsga3.py` |
| MOEA/D (PBI, per-weight sequential update loop) | `moead.py` |
| RVEA (reference-vector adaptation) | `rvea.py` |
| RVEAa (RV regeneration + batch truncation) | `rveaa.py` |
| HypE (Monte-Carlo hypervolume) | `hype.py` |
| Smoke tests | `../../../unit_test/etl/algorithms/mo/` | sibling — write via bash heredoc |
| Parity tests (torch allowed) | `../../../unit_test/etl/algorithms/parity/` | sibling |
| Torch reference | `../../../evox/algorithms/mo/` | sibling — READ-ONLY |

## Escalation (outside this node's write scope)
- `src/evox_etl/CONTEXT.md` "ETL issues found" #1 still claims stale
  "ndarray rejected" comments remain in mo/*.py — RESOLVED for this node
  (T2-A44); root agent should condense that bullet once the SO rounds and
  problems/numerical land. DESIGN.md §4.1 was refreshed upstream (make_*
  policy) and now matches this node's state.
- The EvalMonitor's fixed `(pop_size, dim)` buffers cannot absorb RVEAa's
  (2*n_v, dim) population or its NaN rows (same core-side constraint the
  CoDE family hit); needs a core/workflows-side monitor contract decision.
