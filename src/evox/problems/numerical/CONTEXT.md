# evox/problems/numerical — benchmark problems in functional style

## Intent
The 2.0 numerical benchmark problems (basic, DTLZ, CEC2022). Each problem is a frozen config dataclass + plain module-level `evaluate(config, problem_state, pop) -> (fitness, problem_state)` (PLAIN functions, NO `@evoxc.defn` — DESIGN §4.3). DTLZ adds `pf(config)` reference-Pareto-front functions. Numerical problems are stateless: they share the empty frozen `ProblemState` (`state.py`); the uniform signature is kept anyway.

Not implemented (external-library dependent — reported to root): the brax, mujoco_playground and supervised_learning neuroevolution problems. The virtual Gaussian-noise problem (`../neuroevolution/`) and the host-side `hpo_wrapper` ARE implemented — see `../CONTEXT.md`.

## API Surface
- `basic.py`: 10 configs (`ShiftAffineNumericalProblem`, `Ackley(a=20.0, b=0.2, c=2π)`, `Griewank`, `Rastrigin`, `Rosenbrock`, `Schwefel`, `Sphere`, `Ellipsoid`, `Zakharov`, `Levy` — shift/affine numpy-array fields) + 9 `*_func`s (1.x signatures) + module-level `evaluate` dispatching statically on config type. NO boundary handling (raw math).
- `dtlz.py`: `DTLZ1(d=7)/DTLZ2..6(d=12)/DTLZ7(d=21)` configs (m=3, ref_num=1000) + `evaluate`/`pf` dispatchers → per-variant `evaluate_dtlzN`/`pf_dtlzN` (DTLZ3/DTLZ4 share `pf_dtlz2`). Sampling (Das-Dennis uniform + grid) uses the canonical `evox.operators.sampling` `uniform_sampling`/`grid_sampling` — note they return `(tensor, n_samples)` tuples, call sites unpack `[0]`.
- `cec2022.py`: `CEC2022(problem_number, dimension)` config + `evaluate` + all internal math as plain functions (`cec2022_f1..f12`, `shift/rotate/cut/sr_func_rate/cf_cal`, basic funcs). Imports 5 basic funcs from `basic`.
- `__init__.py`: re-exports all classes, funcs, `CEC2022`, `DTLZ1-7`, `ProblemState`, and the three modules.

## Verified evoxc facts (do not re-investigate)
- `evoxc.evaluate` REJECTS static (non-tensor) args. Use `evoxc.build(fn, cfg, state, spec)` + `evoxc.run(exe, cfg, state, x)` — run() wants ALL positional args (static ones by value); `Executable` has no `.run` method; `TensorSpec` shape is a flat tuple `(n, d)` (no `from_tensor`).
- Config dataclasses holding numpy arrays (basic.py shift/affine) need NO registration on the installed evoxc: ndarray is accepted as a static trace leaf, so the former zero-child `evoxc.register_pytree_node` loop is gone (do not re-add it — a registered config becomes an opaque node that BYPASSES the run-time static-value equality check: a different config value at `evoxc.run` is accepted silently → wrong-answer drift, whereas plain-leaf ndarray fields raise TraceError on mismatch). Plain int-field dataclasses (dtlz/cec2022) work as static args directly.
- `x[:, a:b]` slices FAIL whenever an axis keeps a full-axis `:` over a dynamic dim → use `evoxc.gather(x, const_int32_arange, axis=1)` (np.take semantics, result `(n, len(idx))`). Static int column indexing (`x[:, 0]`) works. DTLZ formulas slice heavily → traced shapes must be fully STATIC (fine: workflow builds specs from concrete tensors; tests must use static-shape specs).
- Reductions directly over a trace-input leaf fail (IR Dim-vs-None inconsistency) → apply an elementwise op first (`x * 1.0`); no `zeros_like`/`full_like` in enp → zero-init via `x * 0.0`.
- `enp.expand_dims`/`reshape` cannot carry dynamic dims → unroll static Python loops where needed (see `katsuura_func`).
- `!=` not overloaded on symbolic tensors → `evoxc.not_equal`. `enp.floor` missing → top-level `evoxc.floor`. Top-level `evoxc.zeros/ones` are CONCRETE creators → use `enp.zeros/ones` inside traces. `float32 ** int64` promotes to float64 → cast exponents to float32. `evoxc.dot` needs rank ≥ 2 → `evoxc.matmul` (numpy semantics). `evoxc.gather` accepts int32 indices.
- Constants: `evoxc.constant(evoxc.tensor(np.asarray(..., dtype=np.float32)))` baked INSIDE evaluate (never closure-captured concrete tensors). When a shift is None, add a zero constant instead (static Python branch).
- **`evoxc.flip` / `evoxc.cumprod` are REJECTED by the compiled backends' StableHLO exporter** (BackendError on iree-llvm-cpu/iree-cuda/xla-cuda). dtlz.py replaces them with `_reverse_axis1` (descending-static-index `evoxc.gather`, numpy take semantics) and `_flip_cumprod_axis1` (static-Python-loop prefix product scan + `_reverse_axis1` + trailing ones) — bit-identical to the flip/cumprod constructions. Do not reintroduce `flip`/`cumprod`.

## Config design (current state)
- Problems keep their validation-only `__post_init__`s as the direct-construction API: basic.py's ten configs each call `_validate_shift_affine` (shift vector / square affine, same dim), cec2022.py's `CEC2022` checks the dimension set.
- The validators and their messages are frozen — asserted by unit tests (`unit_test/evox/problems/test_basic.py` ~L183-195, `test_cec2022.py` ~L83-90).
- The zero-child `evoxc.register_pytree_node` loop (former basic.py tail) is DELETED: ndarray statics are legal on the installed evoxc, and unregistered plain-leaf configs keep the run-time static-value check (drifted config at `evoxc.run` raises TraceError).
- Problems do NOT use the algorithms' `make_*` constructor pattern — that policy lives in DESIGN.md §4.
- Validation stays asserts (AssertionError, frozen messages): `basic._validate_shift_affine` (square affine / 1-D shift / matching dim) and `CEC2022.__post_init__` (dim ∈ {2,10,20}, funcs 6-8 not at D=2).

## Export surface and semantics (current state)
- `basic.py` is name-for-name complete: 10 configs + 9 `*_func`s; there is no `d` field (dimension comes from `pop`), shift is applied first then affine, no boundary handling.
- The configs are independent frozen dataclasses (no subclass chain), so `isinstance(x, ShiftAffineNumericalProblem)` is False for `Ackley` and there are no method overrides — dispatch is `isinstance` on the exact config type (`basic.py:266-286`), and an unknown/base config raises `TypeError`.
- `dtlz.py` has all 7 variants and all 5 PF variants (`pf_dtlz1` = sample/2, `pf_dtlz2` shared by DTLZ3/4, `_pf_dtlz56` shared by DTLZ5/6, `pf_dtlz7`), rows 2926 for DTLZ1-4 and 3000 for DTLZ5/6 at `ref_num=1000, m=3`.
- There is no `DTLZ` base class — only `DTLZ1..7` + the `DTLZConfig` union alias; extending DTLZ means adding a config plus an `evaluate_dtlzN`/`pf_dtlzN` pair and a dispatcher branch.
- `cec2022.py` implements all 12 functions; instance selection is `problem_number`, `FileNotFoundError`/unknown-function `ValueError` are deferred to `evaluate`, and there is no `device` field.
- The export surface is a SUPERSET of the 1.x `numerical.__all__` (25 names): it adds `ShiftAffineNumericalProblem`, `Zakharov`, `Levy`, `zakharov_func`, `levy_func`, `ProblemState`.
- `state.ProblemState` is a concrete empty frozen dataclass, distinct from `evox.core.ProblemState` (which is `Any`); numerical `evaluate`s ignore whatever state they receive.
- Name collision: `basic.levy_func` (w = 1+(x-1)/4) and `cec2022.levy_func` (w = 1+x/4) are different public functions — import explicitly, never star-import both.
- Fully STATIC shapes (batch n AND dim) are required: `basic._shift_affine` builds a `zeros(dim)` constant, `dtlz` indexes `_reverse_axis1`/`enp.ones((n, 1))` with Python ints, cec2022 gathers with static index arrays. `core/workflow.py:410-416` traces from concrete state shapes, so this holds for `StdWorkflow`; a dynamic-dim spec fails.

## Data location (self-contained)
CEC2022 input data ships WITH the package in `cec2022_input_data/` (the 1.x file names, so the data stays comparable): 35 `M_<func>_D<dim>.txt`, 12 `shift_data_<func>.txt`, 6 `shuffle_data_<func>_D<dim>.txt` = 53 files (~880 KB).
`Rand_Seeds.txt` is deliberately absent — `cec2022.py` never reads it.
`cec2022.py` resolves `DATA_DIR = importlib.resources.files("evox.problems.numerical").joinpath("cec2022_input_data")` — package-relative (DESIGN.md §5.9), so it works from a source checkout AND from an installed wheel, with no CWD dependence.
`DATA_DIR` is a `Traversable` (`pathlib.Path` in both source and wheel installs); `_read_floats` uses `is_file()`/`open("r")` on it, so do not assume a `str`/`Path`-only handle when touching this code.
`_load_data` parses those files into a memoized `(M, OShift, SS)` numpy cache; numpy is allowed ONLY for that loading, never in graph code.
Data errors surface at trace time (`evaluate` → `_load_data` → `FileNotFoundError: Cannot open <path> for reading`), not at config construction; the message still starts with `Cannot open` (asserted by `unit_test/evox/problems/test_cec2022.py`).
Packaging: `pyproject.toml` carries `[tool.setuptools.package-data]` `"evox.problems.numerical" = ["cec2022_input_data/*.txt"]` (that glob copies all 53 files into the wheel) plus a matching `MANIFEST.in` `include` line for sdists — keep both when touching packaging.
The sibling test `../../../../unit_test/evox/problems/test_cec2022.py` reuses `cec_module.DATA_DIR` for its shift-point fixture, so it follows the package resolution automatically.

## Validation
Pattern: `evoxc.build` + `evoxc.run` (numpy backend) vs independent float64 numpy references — basic: bit-exact vs numpy versions of the formulas, shift/affine variants within float32 rounding; dtlz: evaluate agreement to a few float32 ulps, pf to 2e-6; cec2022: all 12 functions × dims {2,10,20} within rel 1e-5 of the analytic optima/known values. Unit tests: sibling `../../../../unit_test/evox/problems/` (evoxc-only).
