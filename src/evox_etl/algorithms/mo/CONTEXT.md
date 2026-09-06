# evox_etl/algorithms/mo — functional ports of the torch MO algorithms

## Intent
Plain-function ports of `src/evox/algorithms/mo/` (torch, READ-ONLY reference):
nsga2, nsga3, moead, rvea, rveaa, hype. Each module = frozen config dataclass
(`<Name>Config`), frozen state dataclass (`<Name>State`, tensor leaves only),
and `init/init_ask/init_tell/ask/tell` plain functions (NO `@etl.defn` — see
`../../DESIGN.md` §4.3). Bindings: `../../DESIGN.md` §4-5.

## Contract (all six algorithms)
- `init(config, key) -> state`; `init_ask` returns the FULL population (gen 0
  evaluates the whole pop — verified for all six torch classes, HypE included,
  whose init_step also derives `ref = 1.2 * max(fitness)`); `init_tell(config,
  state, fitness) -> state`; `ask` produces the offspring batch; `tell(config,
  state, fitness) -> state` merges `state.pop` + the ask batch (carried in a
  state field — `offspring` / `off` / `next_generation` — since the binding
  tell signature has no candidates arg) and runs the environmental selection.
- Config fields mirror torch `__init__` exactly minus `device`; `lb`/`ub` numpy
  arrays baked as graph constants (`etl.ops.constant(etl.core.tensor(np.asarray(
  cfg.lb, dtype=np.float32)))`); optional ops are plain function refs, `None` =
  algorithm default, resolved inside functions.
- **Config registration**: every config class is registered as a childless
  pytree node (`etl.register_pytree_node(<Name>Config, _config_flatten,
  _config_unflatten)`, see nsga2.py) so `etl.build(fn, config, ...)` +
  `etl.run(exe, config, ...)` positional passing works. WHY it exists today:
  (a) the original reason — etl v1 rejecting ndarray pytree leaves — is
  OBSOLETE on the installed etl (master @f2f50a7, incl. b8062a9 "accept
  np.ndarray as static trace values"); (b) it is still REQUIRED whenever an
  op field is a non-None callable (functions are NOT static values —
  `_flatten_specs` raises TraceError at the callable leaf); (c) it makes the
  config an opaque static node, so `etl.run` never re-validates config
  values (see audit section below).
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

## Audit — config registration hack, `__post_init__`, construction sites
Design-audit findings (T2-A7) for the refactor toward dumb frozen configs +
functional constructors. All claims verified against the installed etl
(master @f2f50a7) by runtime probes, not just source reading.
### Per-module config surface
Shared field block in all six: `pop_size: int`, `n_objs: int`,
`lb: np.ndarray`, `ub: np.ndarray` (required, no defaults), then optional op
fields — the exact op-field typing differs: `Optional[object]` in nsga2
(58-60), `Optional[Any]` in rveaa (48-50), `Optional[Callable]` in
nsga3/moead/rvea/hype.

| Module | Config class | Extra fields (defaults) | Op fields actually read | Op-field usage |
|---|---|---|---|---|
| nsga2.py | `NSGA2Config` @45 | — | NONE (dead surface; docstring 49-52: kept "for signature parity") | register @76 |
| nsga3.py | `NSGA3Config` @33 | `data_type: Optional[Any] = None` | all three | resolution 146-157; `data_type is bool` init branch 119-123; register @75 |
| moead.py | `MOEADConfig` @29 | — | crossover_op @169, mutation_op @173 | selection_op NEVER (docstring 33-34: torch ignores it too); register @59 |
| rvea.py | `RVEAConfig` @34 | `alpha=2.0`, `fr=0.1`, `max_gen=100` | all three | resolution 146 (crossover), 148 (mutation), 161 (selection); register @63 |
| rveaa.py | `RVEAaConfig` @37 (`frozen=True, eq=False`) | same as RVEAConfig | NONE (dead surface — ask/tell use canonical defaults directly) | register @70 |
| hype.py | `HypEConfig` @29 | `n_sample: int = 10000` | NONE (dead surface) | register @63 |

Dead op fields in nsga2/rveaa/hype are a latent parity DIVERGENCE: the torch
references (`src/evox/algorithms/mo/{nsga2,rveaa,hype}.py`) DO honor non-None
custom ops, the etl ports silently ignore them — a user passing a custom
operator gets torch defaults with no diagnostic.
### The register_pytree_node idiom (quoted, nsga2.py:63-76)
```python
def _config_flatten(config: NSGA2Config):
    """Zero-child flattening: the config travels as one opaque static node."""
    return [], config


def _config_unflatten(config: NSGA2Config, _children) -> NSGA2Config:
    return config


# ETL v1 rejects numpy arrays as static pytree leaves (they are neither
# TensorSpecs nor static Python values), so the config (which holds lb/ub as
# ndarrays) is registered as a childless pytree node carrying the whole
# config as its context — it then passes through etl.build/etl.run untouched.
etl.register_pytree_node(NSGA2Config, _config_flatten, _config_unflatten)
```
The 12 helper lines are byte-identical in all six files (modulo the class
name); the 4-line comment above is now STALE (see next section).
### Why the registration exists TODAY (empirical)
- np.ndarray IS now accepted as a static pytree leaf by the installed etl
  (commit b8062a9 "accept np.ndarray as static trace values",
  `_is_static_value` in etl/trace/_tree.py) — the reason the comment gives is
  obsolete; an unregistered config with int+ndarray+None leaves builds and
  runs fine.
- NON-NONE CALLABLES are still rejected (`TraceError: ... is neither a
  core.TensorSpec nor a static Python value` at the function leaf) — this is
  the only case that still NEEDS registration.
- Zero-child registration makes the config an opaque node: no pytree leaves,
  flatten returns the instance as its own "context"; `etl.run` performs NO
  by-value static revalidation (a different-value config instance is
  accepted silently), whereas on the plain-leaf path a drifted value raises
  `TraceError: graph was specialized on 4 ... run-time argument 9 ... does
  not match`.
- `frozen=True` + ndarray fields = broken `__eq__` (ambiguous truth value)
  and unhashable — the reason RVEAaConfig/RVEAaState set `eq=False`.
### `__post_init__` (only rveaa.py:52-54)
`RVEAaConfig` is the ONLY MO config with `__post_init__`:
```python
def __post_init__(self):
    assert self.lb.shape == self.ub.shape and self.lb.ndim == 1 and self.ub.ndim == 1
    assert self.lb.dtype == self.ub.dtype
```
It mirrors torch `RVEAa.__init__` bound-validating asserts
(src/evox/algorithms/mo/rveaa.py:60-61) and guards the `config.lb.shape[0]`
reads in init/ask; the dtype assert is nearly redundant (all code re-casts
`np.asarray(cfg.lb, dtype=np.float32)` anyway).
Inconsistency: torch `RVEA.__init__` (rvea.py:61-62) holds the identical
asserts yet `RVEAConfig` has NO `__post_init__` — the same invariant is
enforced in one twin port but not the other.
No caller ever trips the asserts: the only RVEAaConfig construction in-repo
is `unit_test/etl/algorithms/mo/test_rveaa.py:38` (matching 1-D float32
arrays).
### Construction call sites today
Smoke tests build configs in `make_config()` with kwargs only, op fields
omitted (→ None/defaults), lb/ub as np.ndarray, and pass the SAME config
instance positionally to both `etl.build(fn, cfg, ...)` and
`etl.run(exe, cfg, ...)`: test_nsga2.py:32, test_nsga3.py:32,
test_moead.py:35, test_rvea.py:38, test_rveaa.py:38, test_hype.py:32.
Parity tests same kwargs style: parity/test_nsga2.py:27, test_nsga3.py:27,
test_moead.py:49.
Benchmarks: `benchmarks/etl_vs_torch/bench_mo.py:207-215` — config class
picked from a `{"NSGA2": NSGA2Config, ...}` map, then
`cfg_cls(pop_size=case.pop_size, n_objs=case.n_obj, lb=lb, ub=ub)` with
`lb = np.full(case.dim, MO_LB, np.float32)`; RVEA/RVEAa/HypE have no
benchmark coverage.
NO in-repo caller passes a non-None operator function; no MO module
constructs another MO config; no `dataclasses.replace` on any config
(only on States, which is fine).
### Smells
- Import-time global side effect: each module calls
  `etl.register_pytree_node` at top level; importing anything that touches
  `evox_etl.algorithms.mo` (its own `__init__` or
  `evox_etl/algorithms/__init__.py`, which eagerly imports mo + so) performs
  SIX registrations, plus ten more from `problems/numerical/basic.py:319-331`.
  Blast radius: the process-global `_PYTREE_NODE_REGISTRY` in etl.core.tree
  (aliased into the trace machinery) — registered node classes become
  containers for EVERY pytree walk in the process; lookup walks the MRO;
  re-registration of the same type silently overwrites; there is no
  unregister API.
- Registration follows each class definition immediately (helper + stale
  comment + register), so there is no import-order hazard today, but the
  side effect is order-sensitive if the pattern is ever reused carelessly.
- 6× duplicated 12-line flatten/unflatten boilerplate + 4-line stale comment
  (cf. the single for-loop over ten types in problems/numerical/basic.py).
- Naming/typing inconsistencies: nsga2 `Optional[object]` vs
  `Optional[Callable]`; rveaa `Optional[Any]`; direct `dataclass` import
  (nsga2/moead/rvea/hype/rveaa) vs `import dataclasses` (nsga3);
  rveaa alone sets `eq=False` on config AND state.
- Config class names differ from torch class names; the aliases
  (`NSGA2 = NSGA2Config`, algorithms/__init__.py:48-53) are the intentional
  parity layer.
### Refactor consequences (replace registration with functional constructors)
- A builder/constructor that normalizes `lb`/`ub` (e.g. float tuples, as the
  SO modules do via tuple-normalizing `__post_init__` — clpso.py:44-52,
  pso.py:43) is NOT even needed for ndarray leaves per se anymore (etl
  accepts ndarray statics) — it IS needed only to keep the config a plain
  static-leaf pytree whose values etl.run re-validates by value.
- The registration stays REQUIRED in exactly one case: non-None callable op
  fields (functions are not static values anywhere in the pytree).
  Refactor options: keep zero-child registration for callable-bearing
  configs (accepting no run-time revalidation), or resolve op choice OUTSIDE
  the config (enum/name resolved host-side, or the builder closes over the
  function at construction and stores only plain statics), or DELETE the
  never-read op fields in nsga2/rveaa/hype and keep only what the algorithm
  reads (nsga3/moead/rvea), documenting the torch-parity intent in the
  docstring instead.
- A constructor-normalization refactor removes `__post_init__` entirely:
  move rveaa's bound validation into the builder/factory (assert once at
  construction on the host), so configs stay dumb frozen dataclasses.
### Escalation-worthy stale records (outside this node)
- `src/evox_etl/CONTEXT.md` "ETL issues found" #1 (ndarray config fields
  rejected by etl.build) no longer reproduces on installed etl — ask the
  root agent to update/retire it (and the SO workarounds it spawned).
- `./src/evox_etl/DESIGN.md` §4.1 documents neither the registration
  convention nor the ndarray-as-static change — needs updating to match
  reality (or the refactored convention).
