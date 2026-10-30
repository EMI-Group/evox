# unit_test/etl/algorithms — algorithm smoke tests (ETL-only)

## Intent
Shared scaffolding + tests for the functional evox_etl algorithm port
(`src/evox_etl/algorithms/`).  No torch imports here — the old torch-parity
suite was removed with the torch package; expectations are hand-verified numpy
references.  The virtual-ES /
VirtualLoRA-ES smoke tests and the virtual end-to-end (real neuroevolution
problem) convergence tests live under `so/es_variants/`.

- `conftest.py` — sys.path shim making the repo root + `src/` importable.
- `helpers.py` — toy problems (Sphere/Rosenbrock/Ackley/DTLZ1) + the generic
  STEP-PROTOCOL driver `run_generations` on the evoxc `"numpy"` backend.
- `test_helpers.py` — tests for the scaffolding itself (pytest).
- `so/de_variants/` — per-variant smoke tests plus the regressions for the
  ported upstream fixes: `test_init_bounds.py` (CoDE/SaDE/SHADE initial
  population uniform in `[lb, ub]`, pinned on a tight `[3, 4]` box) and
  `test_shade.py` (success-history F/CR memory adapts off 0.5 while the run
  still converges). `so/es_variants/` likewise pins the NES default `pop_size`
  (`4 + floor(3 ln dim)`) and GuidedES with `subspace_dims < dim`.
- `test_shim_*.py` — CONVERTED to canonical imports (file names kept for
  history; they no longer import the deprecated `evox_etl.algorithms._shim_*`
  compat stubs):
  - `test_shim_crossover.py` → `evox_etl.operators.crossover`. NOTE canonical
    `DE_differential_sum` returns the first-sampled index as **int32** with
    `replace=True` and **int64** with `replace=False` (the python-int scalar
    in the fix-up `evoxc.select` promotes, torch promotes the same way); the old
    shim cast it back to int32.
  - `test_shim_mutation_sampling.py` → `evox_etl.operators.mutation` /
    `.sampling`; canonical `polynomial_mutation(key, x, lb, ub, pro_m=1.0,
    dis_m=20.0)` takes separate 0-d lb/ub tensors (defaults baked at trace,
    NOT re-passed to `evoxc.run`).
  - `test_shim_selection_basic.py` → `evox_etl.operators.selection`. The old
    shim pre-split the key once (its full-tournament rows luckily always drew
    the global best); canonical operators draw directly from the given key, so
    winner assertions now replicate the internal draws with the same key
    (keys are pure → bit-identical draws) and check exact winners in numpy.
  - `test_shim_selection_nd.py` → `evox_etl.operators.selection` +
    `.selection.non_dominate` (`dominate_relation` is not package-exported,
    mirrors torch).
  - `test_shim_selection_rvea.py` → `evox_etl.operators.selection`
    (`ref_vec_guided`) + `.selection.rvea_selection` (`apd_fn`). apd_fn inputs
    use non-negative partition indices only: canonical gathers `norm_obj` with
    `relu(x)` where torch does `norm_obj[x]` (negative-index wrap) — KNOWN
    latent canonical-operator bug, reported (do not pin in tests).
    `_cosine_similarity` no longer exists canonically (inlined in
    `ref_vec_guided`), so the shim-only helper test was replaced by
    `test_ref_vec_guided_nonorthogonal_vectors` (45deg reference vectors,
    checked against the hand-verified numpy reference).
  - `test_shim_utils.py` → `evox_etl.operators.jit_fix_operator` (canonical
    home for the jit-fix utils, which torch keeps in `evox/utils/`).

## The step-protocol harness (helpers.run_generations) — FINAL API
```python
run_generations(algo_mod, algo_cfg, prob_cfg, n_gens, seed=0) -> final_algorithm_state
```
- `algo_mod` must expose plain functions `init(config, key) -> state` and
  `step(config, state, evaluate) -> state` (the evox_etl step protocol, see
  `src/evox_etl/core/algorithm.py`).
- Dispatch mirrors `StdWorkflow._run_loop`: gen 0 → `init_step` if the module
  defines one (else `step`); the LAST gen → `final_step` if defined (else
  `step`); middle gens → `step`. `n_gens==1` runs only the gen-0 dispatch;
  `n_gens==0` returns the raw init state.
- The `evaluate` closure is built INSIDE the traced generation function,
  closing over the static problem config: `lambda pop: toy_evaluate(prob_cfg,
  ToyProblemState(), pop)[0]`. Algorithm/problem configs are baked as closure
  constants, so each generation graph's ONLY signature input is the algorithm
  state pytree (`evoxc.run(gen_exe, state)`).
- Exe cache is keyed by `(id(step_fn), tensor-spec key of state)` — one build
  per (step function, state shape); NSGA-style first-generation shape changes
  cost one extra build.
- Calibration data for convergence assertions (seed=0): PSO pop40 dim8 on
  Sphere → best ≈ 2.7 after 10 gens, ≈ 1e-2 after 30; SHADE pop16 dim8 →
  ≈ 35 after 5 gens, ≈ 1.5 after 20; CMA-ES pop8 dim5 on Sphere → ≈ 3.7e-3
  after 40 gens; NSGA2 pop20 dim7 3-obj on DTLZ1 → all objective minima < 5
  after 10 gens.
- `evoxc.build` traces a function's Python body once per BUILD (later `evoxc.run`
  calls replay the graph without re-invoking Python) — record builds, not
  calls, when asserting dispatch (see `_recorded` in `test_helpers.py`).

## Constraints
- `evoxc` has no eager mode: all ops inside functions traced via
  `evoxc.build`/`evoxc.run` (backend `"numpy"`).
- `evoxc.run` requires ALL signature args, static dataclasses included — unless
  they are baked into the traced closure (the harness pattern).
- Scalar graph inputs are 0-d numpy arrays; reductions use `axes=`.
