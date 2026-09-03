# unit_test/etl/algorithms — algorithm smoke/parity tests (etl-only)

## Intent
Shared scaffolding + tests for the functional evox_etl algorithm port
(`src/evox_etl/algorithms/`).  No torch imports here — parity tests that
import torch live in `unit_test/etl/parity/`.

- `conftest.py` — sys.path shim making the repo root + `src/` importable.
- `helpers.py` — toy problems (Sphere/Rosenbrock/Ackley/DTLZ1) + generic
  `run_generations` init/ask/tell driver on the etl `"numpy"` backend.
- `test_helpers.py` — tests for the scaffolding itself (pytest).
- `test_shim_*.py` — CONVERTED to canonical imports (file names kept for
  history; they no longer import the deprecated `evox_etl.algorithms._shim_*`
  compat stubs):
  - `test_shim_crossover.py` → `evox_etl.operators.crossover`. NOTE canonical
    `DE_differential_sum` returns the first-sampled index as **int32** with
    `replace=True` and **int64** with `replace=False` (the python-int scalar
    in the fix-up `etl.select` promotes, torch promotes the same way); the old
    shim cast it back to int32.
  - `test_shim_mutation_sampling.py` → `evox_etl.operators.mutation` /
    `.sampling`; canonical `polynomial_mutation(key, x, lb, ub, pro_m=1.0,
    dis_m=20.0)` takes separate 0-d lb/ub tensors (defaults baked at trace,
    NOT re-passed to `etl.run`).
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
  - `test_shim_utils.py` → `evox_etl.algorithms._jit_fix_operator` (staging)
    with a `# TODO:` comment to repoint to `evox_etl.operators.jit_fix_operator`
    once that move lands.

## Constraints
- ETL has no eager mode: all ops inside functions traced via
  `etl.build`/`etl.run` (backend `"numpy"`).
- `etl.run` requires ALL signature args, static dataclasses included.
- Scalar graph inputs are 0-d numpy arrays; reductions use `axes=`.
