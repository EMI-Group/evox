# unit_test/etl/operators — tests for the functional evox_etl operators

## Intent
pytest suite mirroring `src/evox_etl/operators/` (pure functional genetic
operators on ETL). Two kinds:
1. **Property tests** (top level, NO torch imports): determinism (same key +
   inputs → identical output), key divergence, and shape/dtype/bounds/invariant
   contracts for the keyed random operators; determinism + invariants for the
   keyless samplers.
2. **`parity/` — parity tests vs the torch evox reference** (torch imports
   allowed ONLY here): same numpy inputs through `etl.build(fn, *specs,
   backend="numpy")` + `etl.run(exe, *args)` on one side and `torch.tensor`
   inputs on the other, compared at 1e-6 (bool/int outputs exactly).

Run: `/mnt/local-ssd/bchuang/evox/.venv/bin/python -m pytest unit_test/etl/operators -q`
(also collected by the `unit_test/etl` suite run).

## Structure
| File | Operators covered |
|---|---|
| `test_sampling.py` | uniform_sampling, grid_sampling (deterministic); latin_hypercube_sampling_standard, latin_hypercube_sampling (keyed) |
| `test_selection.py` | tournament_selection, tournament_selection_multifit, select_rand_pbest (keyed random only) |
| `test_crossover.py` | DE_differential_sum, DE_binary_crossover, DE_exponential_crossover, simulated_binary, simulated_binary_half |
| `test_mutation.py` | polynomial_mutation |
| `parity/test_parity_sampling.py` | uniform_sampling (incl. h2 branch), grid_sampling |
| `parity/test_parity_selection.py` | dominate_relation, non_dominate_rank, crowding_distance, nd_environmental_selection, apd_fn, ref_vec_guided |
| `parity/test_parity_crossover.py` | DE_arithmetic_recombination (K scalar/(pop,)/(pop,1)) |
| `parity/conftest.py` | idempotent sys.path shim (repo root found via pyproject.toml marker) |

`__init__.py` in both `operators/` and `operators/parity/` makes each a
distinct pytest package (no basename collisions when suites run together).

## Test strategy notes
- Statics passed to `etl.build` are re-passed to `etl.run` in signature order
  (same value); Python defaults baked at build are omitted at run (issue 17).
  Graphs specialize on static values → fresh executable per distinct static.
- `etl.run` returns etl tensors → convert with `.numpy()` (never
  `np.asarray`, issue 16). Tuple-returning functions give tuples of etl
  tensors. Keys are 0-d int64 ndarrays (`TensorSpec((), "int64")`).
- Int dtypes: tournament/multifit/select outputs int32; non_dominate_rank
  int32; DE_differential_sum second output int64 with `replace=False` (the
  select against the python-int mirror promotes the int32 randint draws) and
  int32 with `replace=True` — asserted exactly.
- Known deviations (operator mirrors observable torch behaviour; reported,
  not fixed — see `src/evox_etl/operators/` CONTEXTs):
  - `latin_hypercube_sampling_standard(smooth=False)` is NOT row-identical
    across keys (the cell-permutation draw always happens, as in torch); the
    deterministic multiset of cell centers per column IS asserted.
  - `apd_fn` masked entries (x == -1): etl indexes `norm_obj` via relu(x)
    (→ row 0), torch via raw x (negative wrap-around → last row);
    `ref_vec_guided` overwrites exactly those entries with inf, so its
    observable output is bit-identical — the parity test compares apd_fn's
    unmasked entries only, plus a fully-assigned (x >= 0) case.
  - `ref_vec_guided` returns NaN rows for empty reference-vector partitions —
    faithful torch parity, compared with `equal_nan=True`.
  - SBX "children within parent range" is deliberately NOT tested (beta > 1
    extrapolates, same as torch); the valid invariant `c1 + c2 == p1 + p2` is.
- Torch `non_dominate_rank` is called eagerly (never under torch.compile).
- Keep files focused (<~400 lines); runtime moderate (pop 32–64, dim 8–20,
  n_round ~100); build executables once per test and reuse.
