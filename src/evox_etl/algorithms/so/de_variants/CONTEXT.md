# evox_etl/algorithms/so/de_variants — functional DE-family ports

## Intent
Plain-function step-protocol ports of the torch evox DE family (READ-ONLY
reference: `../../../../evox/algorithms/so/de_variants/`), one module per torch
file: `de.py`, `jade.py`, `shade.py`, `sade.py`, `ode.py`, `code.py`.
Binding spec: `../../DESIGN.md` (read fully before writing code) and the
protocol doc `../../core/algorithm.py`.

## API Surface
- Each module: frozen config dataclass named after the torch class (`DE`, `JaDE`,
  `SHADE`, `SaDE`, `ODE`, `CoDE`) + frozen `<Name>State` dataclass + plain
  functions `init(config, key) -> state` and
  `step(config, state, evaluate) -> state` owning ONE full generation
  (old ask-body → `fitness = evaluate(candidates)` → old tell-body, fused).
- Each config module also defines a `make_*` constructor in the SAME module
  (`make_de`, `make_jade`, `make_shade`, `make_sade`, `make_ode`, `make_code`);
  `__init__.py` exports the classes and all 6 `make_*`.
- `init_step(config, state, evaluate) -> state` exists ONLY on `de.py`, `ode.py`,
  `jade.py` (their torch counterparts define `init_step`: evaluate the initial
  population). The torch SHADE/SaDE/CoDE have no init_step, so those modules
  define none and the workflow falls back to `step` for generation 0.
- No `final_step` anywhere (the torch reference defines no algorithm-level
  overrides). `ask`/`tell`/`init_ask`/`init_tell` are GONE.
- State leaves are etl tensors ONLY (float32 preferred, int32/int64 indices, key `()`
  int64) and their dtypes/shapes must be STABLE across generations (compile-once
  step graphs — no per-run dtype drift, see the SaDE cast notes below).
- `step` never re-threads `evaluate`; ODE calls it TWICE per generation (DE
  trials, then the opposition population `lb + ub - pop` built from the
  POST-selection pop — one etl step == one torch ODE step; the old two-phase
  state machine and the `opposition`/`phase` state fields are gone).

## Constraints
- Import crossover/sampling/selection operators from `evox_etl.operators`
  (canonical, torch-parity-verified) and the jit-fix utils (clamp, clamp_float,
  `_take_along_axis`, ...) from `evox_etl.operators.jit_fix_operator`.
- No torch/numpy in graph code (numpy only for baking constants via
  `etl.ops.constant(etl.core.tensor(np.asarray(...)))`).
- Files < ~400 lines; static Python loops/config branches allowed in traces.

## Notes for agents (verified — do not re-investigate)
- **Configs are dumb frozen dataclasses** (no `__post_init__`, no normalization,
  no validation) storing ONLY Python scalars + flat float tuples (`lb`/`ub`/
  `mean`/`stdev`/`differential_weight`).
- Construct configs via the module-level `make_*` constructor (normalizes
  arrays→plain float tuples and validates with ValueError naming the param);
  direct construction with already-normalized statics stays legal.
  Installed etl accepts np.ndarray static values, so raw-array fields trace
  fine, but `make_*` is the sanctioned entry point (policy: `DESIGN.md` §4.1).
  CoDE `param_pool` stores a nested tuple of (F, CR) pairs (natural (3, 2)
  shape — flattened storage would break the trial generation's gather over
  param_ids).
- Normalization/baking helpers are consolidated in
  `evox_etl/algorithms/_config_utils.py` (`to_float_tuple` dtype-preserving,
  `normalize_bounds`, `require_ge`/`require_between`/`require_choice`,
  `bake_float32_constant`, `bake_bounds(..., as_row=True)`); this family no
  longer defines local `_to_float_tuple`/`_bounds`/`_bake*` helpers.
- **`etl.select` does NOT numpy-broadcast** a `(n,)` condition against `(n, m)`
  operands (ShapeError). Always `enp.expand_dims(cond, 1)` first. Scalar conds
  broadcast fine. Python float + float32 → float32 OK; Python int + int32 → int64
  (avoid — use `enp.full/ones/zeros/arange` with explicit dtype).
- `etl.gather` = numpy take semantics; a 0-d index drops the indexed axis
  (result `(dim,)`). Row-local gather needs `_take_along_axis` (2-D only).
- RNG: `key, subkey = random.split(state.key)` at function entry; one split per
  random op, in torch's draw order; store the advanced `key` back in the state.
  The post-evaluate halves draw no randomness. `random.multinomial(key, probs, n)`
  returns int32.
- Bounds baked per function via `bake_bounds(lb, ub, as_row=True)` (shared
  helper — `(1, dim)` float32 constants) or `bake_float32_constant(x, shape=(1, -1))`
  for a single bound; `dim = len(config.lb)` stays a static Python int (works for
  both tuple and ndarray fields).
- SHADE/SaDE/CoDE init populations use **`randn` scaled by bounds** (torch quirk —
  no uniform, no clamp) — do not assert in-bounds on their initial pops.
- `etl.median(x, axis=0)` matches torch's NaN propagation BUT promotes
  float32→float64; `int32 * bool` also promotes to int64 — cast back explicitly
  (SaDE does; see the casts around `CRM_update`/`success_counts`/
  `failure_counts` there). Compile-once step graphs REQUIRE stable state
  dtypes; the old ask/tell harness masked this by rebuilding exes per
  generation.
- `etl.roll(x, shift, axis)`; `etl.nansum(x, axes=0)`; `etl.argsort(x, axis=0,
  stable=True)` (int64 indices).
- Verified host pattern (dataclass config as static arg + dataclass state pytree):
  `etl.build(fn, cfg, spec_tree, backend="numpy")` / `etl.run(exe, cfg, state)` —
  wrap `step` in a `body(cfg, state) -> step(cfg, state, evaluate)` closure with
  the toy `evaluate` inlined. See `unit_test/etl/algorithms/helpers.py` for the
  build/run mechanics.
- Parity tests MUST use a unique basename (`test_<name>_parity.py`, not
  `test_<name>.py`): pytest's prepend import mode collides when the same basename
  exists in `so/de_variants/` and `parity/` and both are collected in one run.
- The DE parity test uses the MEDIAN of 3 seeds with the 10% margin: per-seed
  best fitnesses fluctuate ~0.57-1.92× around torch (different RNG streams), so a
  single-seed 10% margin fails ~1/3 of the time by chance. Median-of-3 is stable.
- **ode.py imports `_de_trial` from de.py** (algorithm logic — stays; the
  mutation/crossover helper is part of ODE's API) and the shared baking helpers
  from `algorithms/_config_utils.py`; keep ode.py compiling when touching either.
  code.py factors its candidate generation into `_generate_trials(config, state)
  -> (trial_batch, advanced_key)`.
- Legacy direct-dataclass-construction sites (family smoke tests, parity
  test_de_parity.py:45, benchmarks/etl_vs_torch/bench_so.py:115) still pass raw
  np.ndarray `lb`/`ub` and still trace (bakers accept ArrayLike; etl accepts
  ndarray statics); they should migrate to `make_*` (parallel migration round).
  Tests only exercise default hyperparameters — the tuple `differential_weight`
  (ndv>1), `mean`/`stdev` init, and non-default `param_pool` paths have NO test
  coverage.
- KNOWN ISSUE (pre-existing, NOT introduced by the step-protocol conversion):
  DE/ODE with `num_difference_vectors > 1` + tuple `differential_weight`
  constructs and validates fine but FAILS at trace time in `_de_trial`
  (`base + f_const * difference_vector`, ShapeError — cannot broadcast ndv
  against dim): the (ndv,) weight constant never multiplies per-individual
  difference vectors. Reproduced identically on the pre-conversion HEAD.
- KNOWN ISSUE (core/workflows, NOT this family): `EvalMonitor`'s fixed
  `(pop_size, dim)` monitor buffers cannot absorb CoDE's torch-faithful
  `(3*pop_size, dim)` candidate batch under the compile-once step protocol —
  StdWorkflow(CoDE, ..., monitor=EvalMonitor) raises ShapeError at the second
  generation. CoDE itself (no monitor) is fine. Needs a core-side fix
  (dynamically-sized monitor state or a fixed-size monitor contract).

## Routing Table
| Area | Path |
|---|---|
| DE + ODE (shared mutation/crossover helpers) | `de.py`, `ode.py` |
| JaDE / SHADE / SaDE / CoDE | `jade.py`, `shade.py`, `sade.py`, `code.py` |
| Smoke tests (no torch) | `../../../../unit_test/etl/algorithms/so/de_variants/` (sibling — write via bash heredoc) |
| Parity test (torch allowed) | `../../../../unit_test/etl/algorithms/parity/test_de_parity.py` (sibling) |
| Canonical operators (import from) | `../../../operators/crossover`, `../../../operators/selection`, `../../../operators/jit_fix_operator.py` |
| Step-protocol contract | `../../core/algorithm.py` (protocol doc), `../../core/workflow.py` (`_make_step_fn`) |
