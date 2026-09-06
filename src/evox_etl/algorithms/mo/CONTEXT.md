# evox_etl/algorithms/mo — functional ports of the torch MO algorithms

## Intent
Plain-function ports of `src/evox/algorithms/mo/` (torch, READ-ONLY reference):
nsga2, nsga3, moead, rvea, rveaa, hype. Each module = frozen config dataclass
(`<Name>Config`), frozen state dataclass (`<Name>State`, tensor leaves only),
and `init/init_ask/init_tell/ask/tell` plain functions (NO `@etl.defn` — see
`../../DESIGN.md` §4.3). Bindings: `../../DESIGN.md` §4-5.

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
- Op fields: nsga3/moead/rvea KEEP `Optional[Callable]` op fields (genuinely
  read — nsga3 all three + `data_type: Optional[Any] = None`; moead
  `mutation_op`/`crossover_op`, `selection_op` accepted-and-ignored because
  torch ignores it too; rvea all three + `alpha=2.0`, `fr=0.1`,
  `max_gen=100`) and KEEP zero-child `etl.register_pytree_node`
  registration — REQUIRED there because non-None callables are NOT static
  values anywhere in the pytree (TraceError at the callable leaf).
  Registration makes the config an opaque static node: `etl.run` performs NO
  by-value revalidation.
- nsga2/rveaa/hype DELETED their never-read `selection_op`/`mutation_op`/
  `crossover_op` fields AND their registration; class docstrings note the
  torch signature-parity intent (the torch classes DO honor custom ops; the
  etl functional variants hard-code tournament_selection_multifit /
  simulated_binary[_half] / polynomial_mutation). These configs are now
  plain static-leaf pytrees, revalidated by value at `etl.run`.
- `make_*` signatures (drives the test-migration round; all raise ValueError
  on malformed bounds; the three callable-bearing ones also raise ValueError
  on a non-callable non-None op field):
  - `make_nsga2(pop_size: int, n_objs: int, lb: ArrayLike, ub: ArrayLike) -> NSGA2Config`
  - `make_nsga3(pop_size: int, n_objs: int, lb: ArrayLike, ub: ArrayLike, selection_op: Optional[Callable] = None, mutation_op: Optional[Callable] = None, crossover_op: Optional[Callable] = None, data_type: Optional[Any] = None) -> NSGA3Config`
  - `make_moead(pop_size: int, n_objs: int, lb: ArrayLike, ub: ArrayLike, selection_op: Optional[Callable] = None, mutation_op: Optional[Callable] = None, crossover_op: Optional[Callable] = None) -> MOEADConfig`
  - `make_rvea(pop_size: int, n_objs: int, lb: ArrayLike, ub: ArrayLike, alpha: float = 2.0, fr: float = 0.1, max_gen: int = 100, selection_op: Optional[Callable] = None, mutation_op: Optional[Callable] = None, crossover_op: Optional[Callable] = None) -> RVEAConfig`
  - `make_rveaa(pop_size: int, n_objs: int, lb: ArrayLike, ub: ArrayLike, alpha: float = 2.0, fr: float = 0.1, max_gen: int = 100) -> RVEAaConfig`
  - `make_hype(pop_size: int, n_objs: int, lb: ArrayLike, ub: ArrayLike, n_sample: int = 10000) -> HypEConfig`
- Direct `*Config(...)` dataclass construction with ndarray lb/ub still runs
  (field annotations are unenforced and the installed etl accepts ndarray
  statics) but is not the sanctioned API — unit-test files are being
  migrated to `make_*` by a parallel test-migration round.

## Contract (all six algorithms)
- `init(config, key) -> state`; `init_ask` returns the FULL population (gen 0
  evaluates the whole pop — verified for all six torch classes, HypE included,
  whose init_step also derives `ref = 1.2 * max(fitness)`); `init_tell(config,
  state, fitness) -> state`; `ask` produces the offspring batch; `tell(config,
  state, fitness) -> state` merges `state.pop` + the ask batch (carried in a
  state field — `offspring` / `off` / `next_generation` — since the binding
  tell signature has no candidates arg) and runs the environmental selection.
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
  → `random.split_n(key, n)`; advanced `key` stored back. Ask/tell deterministic
  given state.
- Effective pop_size: MOEAD/RVEA/RVEAa overwrite torch `self.pop_size` with the
  Das-Dennis count `n_v` from `uniform_sampling(pop_size, n_objs)` (returns
  `(points, n_samples)` — the static int drives shapes). NSGA2/NSGA3/HypE keep
  the user pop_size. SBX quirk preserved: offspring batch = 2*(n_v//2) rows
  (torch `x[n//2 : n//2*2]`), e.g. 14 for n_v=15.
- RVEA: pop stays (n_v, dim); RVEAa: after the first tell pop GROWS to
  (2*n_v, dim) (one survivor row per reference vector, NaN rows possible for
  unmatched vectors). Both: mating pool always draws the FIXED n_v, and the
  torch `_mating_pool` arange/sorted_indices spans `pop.shape[0]` rows — never
  derive n_v from `pop.shape[0]`.
- NSGA3 tell's linalg is decomposed into exporter-safe ops (the stablehlo-v1
  exporter DEFERS `matrix_rank`/`svd`/`solve` → BackendError — escalated to
  the root agent): the torch `matrix_rank(extreme) == n_objs` guard becomes
  rank = count(sqrt(eigvalsh(AᵀA)) > s_max·n·eps32) with AᵀA accumulated in
  float64 (noise floor ~1e-8·s_max « cutoff ~1e-7·s_max; verified ≡
  `etl.matrix_rank` on 14k random/singular/near-singular matrices), and the
  torch `solve(extreme, ones)` hyperplane becomes the eigh-based normal
  equations (AᵀA)⁻¹Aᵀ·1 in float64 (≤1.2e-7 rel vs the numpy LU solve even
  at cond 1e5). Both build+run on iree-llvm-cpu and xla-cuda.

## ETL gotchas (verified — do not re-investigate)
- MOEAD tell's while_loop: z is a loop carry updated PER-i BEFORE the PBI
  comparisons (torch order — do NOT pre-lower z with the batch min, that
  contaminates the comparisons). cond/body take the carry as ONE tuple arg;
  closure-capture of outer args (`state.w`, `state.next_parents`, `fitness`)
  is legal. Carry `i` must stay int32 (`etl.cast(i + 1, etl.int32)`).
  MOEAD sequential overwrites: the same row can be updated by several i's in
  one tell (last-i-wins) — torch does the same; do not "deduplicate".
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
