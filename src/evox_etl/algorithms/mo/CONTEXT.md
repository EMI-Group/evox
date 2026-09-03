# evox_etl/algorithms/mo — functional ports of the torch MO algorithms

## Intent
Plain-function ports of `src/evox/algorithms/mo/` (torch, READ-ONLY reference):
nsga2, nsga3, moead, rvea, rveaa, hype. Each module = frozen config dataclass
(`<Name>Config`), frozen state dataclass (`<Name>State`, tensor leaves only),
and `init/init_ask/init_tell/ask/tell` plain functions (NO `@etl.defn` — see
`../../DESIGN.md` §4.3). Bindings: `../../DESIGN.md` §4-5.

## Contract (all six algorithms)
- `init(config, key) -> state`; `init_ask(config, state) -> (candidates, state)`
  returns the FULL population (gen 0 evaluates the whole pop — verified for all
  six torch classes, HypE included, whose init_step also derives `ref` from the
  initial fitness); `init_tell(config, state, fitness) -> state`; `ask` produces
  the offspring batch — torch semantics: FULL `pop_size` offspring per gen for
  NSGA2/NSGA3/RVEA/RVEAa/HypE (selection runs pop_size rounds, SBX returns both
  children per pair) and one offspring per weight vector for MOEAD (pop_size =
  Das-Dennis count there). `tell(config, state, fitness) -> state`.
- Config fields mirror torch `__init__` exactly minus `device`; `lb`/`ub` numpy
  arrays baked as graph constants (`etl.ops.constant(etl.core.tensor(np.asarray(
  cfg.lb, dtype=np.float32)))`); optional ops are plain function refs, `None` =
  algorithm default, resolved inside functions.
- `tell(config, state, fitness)` needs the offspring for the pop/fit merge, so
  `ask` stores its batch in an extra state field (e.g. `offspring`, set by ask,
  consumed by tell) — the candidates-in-state pattern of the DE ports. NOTE:
  np.ndarray config leaves make the config unpassable to `etl.build` directly
  (TraceError: not a static value) — callers partialize the config.
- RVEA/RVEAa: after the first tell `pop` GROWS to (2*n_v, dim) (the survivor
  tensor keeps one row per reference vector, NaN rows included), while
  `reference_vector` stays (2*n_v, m) and the mating pool always draws the
  FIXED Das-Dennis count `n_v = reference_vector.shape[0] // 2` (torch
  `self.pop_size`) — never derive n_v from `pop.shape[0]`; the torch
  `_mating_pool` `arange`/sorted_indices however spans `pop.shape[0]` rows.
- Bounds for the mutation shim: `boundary = enp.stack([lb, ub], axis=0)` —
  shim `polynomial_mutation(key, x, boundary, pro_m, dis_m)` (torch takes lb/ub
  separately).
- RNG: `key, subkey = random.split(state.key)` (returns TWO keys); several draws
  → `random.split_n(key, n)`; advanced `key` stored back. Ask/tell deterministic
  given state.
- Effective pop_size: MOEAD/RVEA/RVEAa overwrite torch `self.pop_size` with the
  Das-Dennis count from `uniform_sampling(pop_size, n_objs)` (returns
  `(points, n_samples)` — the static int is usable at trace time for shapes).
  NSGA2/NSGA3/HypE keep the user pop_size.

## ETL gotchas (verified — do not re-investigate)
- `etl.sum/mean` take `axes=`; `etl.min(x, axes=..)` values only (argmin
  separate); `etl.topk(x, k, axis, largest=False)` → `(values, indices)`;
  `etl.sort(x, axis, descending=, stable=)` values only; `etl.argsort(..., stable=
  True)` → int64 indices. `random.permutation(key, n, dtype=int32)`.
- `etl.cond(pred, true_fn, false_fn, *operands)` — use it for the NSGA3
  hyperplane solve (numpy `linalg.solve` RAISES on singular matrices; never
  compute it unconditionally).
- No boolean/advanced indexing with dynamic masks (no dynamic shapes): select
  survivors via sort/argsort of masked values + static `[:pop_size]` slice, or
  one-hot `reduce_max(equal(...))` masks for torch `_masked_assign` patterns.
- `etl.gather` = numpy `take`: row-local gathers → `_take_along_axis` from
  `evox_etl.algorithms._shim_selection_basic`; `etl.scatter` = put_along_axis
  replacement (no scatter-add — torch NSGA3 `scatter_add` → one-hot sum).
- No `&`/`!=`/`%` overloads → `enp.logical_and`, `etl.not_equal`,
  `etl.remainder`. `enp.zeros/enp.full` (not `etl.zeros`); Python-int tensor
  promotion → wrap in `etl.cast`. `etl.stack` of mixed dtypes is risky → cast
  int32 rank to float32 before stacking in `tournament_selection_multifit`.
- while_loop carry is ONE pytree argument (unpack inside cond/body); carry
  dtypes must be identical across iterations.
- No `etl.cdist` → MOEAD neighbors via
  `sqrt(maximum(d2, 0))`, `d2 = w2[:, None] + w2[None, :] - 2*dot(w, w.T)`.

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
