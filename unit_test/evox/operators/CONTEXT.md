# unit_test/evox/operators — tests for the functional evox operators

## Intent
pytest suite mirroring `src/evox/operators/` (pure functional genetic
operators on evoxc): determinism (same key + inputs → identical output), key
divergence, and shape/dtype/bounds/invariant contracts for the keyed random
operators; determinism + invariants for the keyless samplers.
Every test drives an operator through `evoxc.build(fn, *specs,
backend="numpy")` + `evoxc.run(exe, *args)` with numpy inputs — torch is not a
dependency of this repository.

Run: `PYTHONPATH=src <venv>/bin/python -m pytest unit_test/evox/operators -q`
(86 tests, ~5s; also collected by the `unit_test/evox` suite run).

## Structure
| File | Operators covered |
|---|---|
| `test_sampling.py` | uniform_sampling, grid_sampling (deterministic); latin_hypercube_sampling_standard, latin_hypercube_sampling (keyed) |
| `test_selection.py` | tournament_selection, tournament_selection_multifit, select_rand_pbest (keyed random only); incl. the 1-D `(n_round,)` shape regressions for `n_round` 1/3 and the very-large-tournament argmin / lexicographic-winner regressions |
| `test_crossover.py` | DE_differential_sum, DE_binary_crossover, DE_exponential_crossover, simulated_binary, simulated_binary_half; incl. the uniform-mask regression (empirical crossover rate `CR + (1-CR)/dim`, a normal-draw mask fails both the CR=0.1 and CR=0.9 bands) |
| `test_mutation.py` | polynomial_mutation |
| `test_jit_fix_operator.py` | switch, clip, maximum_float, minimum_float (property tests: the trace-only jit-fix helpers) |

`__init__.py` in `operators/` makes it a distinct pytest package (no basename
collisions when suites run together).

## Test strategy notes
- Statics passed to `evoxc.build` are re-passed to `evoxc.run` in signature order
  (same value); Python defaults baked at build are omitted at run (issue 17).
  Graphs specialize on static values → fresh executable per distinct static.
- `evoxc.run` returns evoxc tensors → convert with `.numpy()` (never
  `np.asarray`, issue 16). Tuple-returning functions give tuples of evoxc
  tensors. Keys are 0-d int64 ndarrays (`TensorSpec((), "int64")`).
- Int dtypes: tournament/multifit/select outputs int32; non_dominate_rank
  int32; DE_differential_sum second output int64 with `replace=False` (the
  select against the python-int mirror promotes the int32 randint draws) and
  int32 with `replace=True` — asserted exactly.
- Known deviations (operator mirrors the observable reference behaviour;
  reported, not fixed — see `src/evox/operators/` CONTEXTs):
  - `latin_hypercube_sampling_standard(smooth=False)` is NOT row-identical
    across keys (the cell-permutation draw always happens, as upstream); the
    deterministic multiset of cell centers per column IS asserted.
  - `apd_fn` masked entries (x == -1): evoxc indexes `norm_obj` via relu(x)
    (→ row 0), the reference indexes with raw x (negative wrap-around → last
    row); `ref_vec_guided` overwrites exactly those entries with inf, so its
    observable output is bit-identical — the masked entries are not asserted,
    a fully-assigned (x >= 0) case is.
  - `ref_vec_guided` returns NaN rows for empty reference-vector partitions —
    faithful to the reference formula, asserted with `equal_nan=True`.
  - SBX "children within parent range" is deliberately NOT tested (beta > 1
    extrapolates, as upstream); the valid invariant `c1 + c2 == p1 + p2` is.
- Keep files focused (<~400 lines); runtime moderate (pop 32–64, dim 8–20,
  n_round ~100); build executables once per test and reuse.
- Regression tests pinning ported upstream fixes live next to their
  siblings: the `DE_binary_crossover` UNIFORM mask draw
  (`pop_size=2000, dim=20`, rate `CR + (1-CR)/dim` within ±0.02, plus
  `CR=0` mutating exactly one gene per row) and the 1-D tournament index
  shape for `n_round` 1 and 3 together with the large-tournament
  argmin / `np.lexsort` lexicographic-winner assertions.

## Constraints
- No torch here: expectations are plain numpy, computed by hand or by an
  independent numpy reference implementation.
