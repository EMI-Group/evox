# evox_etl/problems/numerical — benchmark problems in functional style

## Intent
Functional-ETL port of the torch evox numerical problems (`../../../evox/problems/numerical/`, read-only reference), math mirrored 1:1. Each problem is a frozen config dataclass + plain module-level `evaluate(config, problem_state, pop) -> (fitness, problem_state)` (PLAIN functions, NO `@evoxc.defn` — DESIGN §4.3). DTLZ adds `pf(config)` reference-Pareto-front functions. Numerical problems are stateless: they share the empty frozen `ProblemState` (`state.py`); the uniform signature is kept anyway.

NOT ported (external-library dependent — reported to root): the brax, mujoco_playground and supervised_learning neuroevolution problems. The virtual Gaussian-noise problem (`../neuroevolution/`) and the host-side `hpo_wrapper` ARE ported — see `../CONTEXT.md`.

## API Surface
- `basic.py`: 10 configs (`ShiftAffineNumericalProblem`, `Ackley(a=20.0, b=0.2, c=2π)`, `Griewank`, `Rastrigin`, `Rosenbrock`, `Schwefel`, `Sphere`, `Ellipsoid`, `Zakharov`, `Levy` — shift/affine numpy-array fields) + 9 `*_func`s (torch signatures) + module-level `evaluate` dispatching statically on config type. NO boundary handling (raw math — matches torch).
- `dtlz.py`: `DTLZ1(d=7)/DTLZ2..6(d=12)/DTLZ7(d=21)` configs (m=3, ref_num=1000) + `evaluate`/`pf` dispatchers → per-variant `evaluate_dtlzN`/`pf_dtlzN` (DTLZ3/DTLZ4 share `pf_dtlz2`, mirroring torch inheritance). Sampling (Das-Dennis uniform + grid) uses the canonical `evox_etl.operators.sampling` `uniform_sampling`/`grid_sampling` — note they return `(tensor, n_samples)` tuples, call sites unpack `[0]`.
- `cec2022.py`: `CEC2022(problem_number, dimension)` config + `evaluate` + all internal math as plain functions (`cec2022_f1..f12`, `shift/rotate/cut/sr_func_rate/cf_cal`, basic funcs). Imports 5 basic funcs from `basic`.
- `__init__.py`: re-exports all classes, funcs, `CEC2022`, `DTLZ1-7`, `ProblemState`, and the three modules.

## Verified evoxc facts (do not re-investigate)
- `evoxc.evaluate` REJECTS static (non-tensor) args. Use `evoxc.build(fn, cfg, state, spec)` + `evoxc.run(exe, cfg, state, x)` — run() wants ALL positional args (static ones by value); `Executable` has no `.run` method; `TensorSpec` shape is a flat tuple `(n, d)` (no `from_tensor`).
- Config dataclasses holding numpy arrays (basic.py shift/affine) need NO registration on the installed evoxc (master f2f50a7, numpy backend): ndarray is accepted as a static trace leaf, so the former zero-child `evoxc.register_pytree_node` loop is gone (do not re-add it — a registered config becomes an opaque node that BYPASSES the run-time static-value equality check: a different config value at `evoxc.run` is accepted silently → wrong-answer drift, whereas plain-leaf ndarray fields raise TraceError on mismatch). Plain int-field dataclasses (dtlz/cec2022) work as static args directly.
- `x[:, a:b]` slices FAIL whenever an axis keeps a full-axis `:` over a dynamic dim → use `evoxc.gather(x, const_int32_arange, axis=1)` (np.take semantics, result `(n, len(idx))`). Static int column indexing (`x[:, 0]`) works. DTLZ formulas slice heavily → traced shapes must be fully STATIC (fine: workflow builds specs from concrete tensors; tests must use static-shape specs).
- Reductions directly over a trace-input leaf fail (IR Dim-vs-None inconsistency) → apply an elementwise op first (`x * 1.0`); no `zeros_like`/`full_like` in enp → zero-init via `x * 0.0`.
- `enp.expand_dims`/`reshape` cannot carry dynamic dims → unroll static Python loops where needed (see `katsuura_func`).
- `!=` not overloaded on symbolic tensors → `evoxc.not_equal`. `enp.floor` missing → top-level `evoxc.floor`. Top-level `evoxc.zeros/ones` are CONCRETE creators → use `enp.zeros/ones` inside traces. `float32 ** int64` promotes to float64 → cast exponents to float32. `evoxc.dot` needs rank ≥ 2 → `evoxc.matmul` (numpy semantics). `evoxc.gather` accepts int32 indices.
- Constants: `evoxc.constant(evoxc.tensor(np.asarray(..., dtype=np.float32)))` baked INSIDE evaluate (never closure-captured concrete tensors). When a shift is None, add a zero constant instead (static Python branch).
- **`evoxc.flip` / `evoxc.cumprod` are REJECTED by the compiled backends' StableHLO exporter** (BackendError on iree-llvm-cpu/iree-cuda/xla-cuda). dtlz.py replaces them with `_reverse_axis1` (descending-static-index `evoxc.gather`, numpy take semantics) and `_flip_cumprod_axis1` (static-Python-loop prefix product scan + `_reverse_axis1` + trailing ones) — bit-identical to the flip/cumprod constructions. Do not reintroduce `flip`/`cumprod`.

## Config design (current state)
- Problems keep their validation-only `__post_init__`s as the direct-construction API: basic.py's ten configs each call `_validate_shift_affine` (shift vector / square affine, same dim), cec2022.py's `CEC2022` checks the dimension set.
- The validators and their messages are frozen — asserted by unit tests (`unit_test/etl/problems/test_basic.py` ~L183-195, `test_cec2022.py` ~L83-90).
- The zero-child `evoxc.register_pytree_node` loop (former basic.py tail) is DELETED: ndarray statics are legal on the installed evoxc, and unregistered plain-leaf configs keep the run-time static-value check (drifted config at `evoxc.run` raises TraceError).
- Problems do NOT use the algorithms' `make_*` constructor pattern — that policy lives in DESIGN.md §4.

## Data location (release-blocking)
cec2022 input data is read from the torch tree, NOT shipped with the port: `cec2022.py:27` sets `DATA_DIR = Path(__file__).resolve().parents[4] / "src" / "evox" / "problems" / "numerical" / "cec2022_input_data"`, and `_load_data` parses the 54 tracked `.txt` files (`../../../evox/problems/numerical/cec2022_input_data/`) into a memoized `(M, OShift, SS)` numpy cache.
numpy is allowed ONLY for that loading; never in graph code.
That path breaks in both release scenarios: (a) deleting `src/evox/` deletes the directory, so source checkouts and `../../../../unit_test/etl/problems/test_cec2022.py` (which recomputes the same expression, no skip guard) fail; (b) in any wheel/sdist install it resolves to `<venv>/lib/pythonX.Y/src/evox/...` because `parents[4]` of `site-packages/evox_etl/problems/numerical/cec2022.py` is the stdlib dir.
Nothing under `src/evox_etl/` contains data (`git ls-files` shows no `.txt`/`.json` there), `MANIFEST.in:2` lists only the torch path, and `pyproject.toml` has no `[tool.setuptools]`/`package-data` section; DESIGN.md §5.9 requires port-owned data loaded via `importlib.resources`.
Fix shape: copy the 54 files into this directory, resolve via `importlib.resources`, and add a packaging entry.
Data errors surface at trace time (`evaluate` → `_load_data`), not at config construction.

## Validation
Pattern: `evoxc.build` + `evoxc.run` (numpy backend) vs references — basic: bit-exact vs numpy versions of the torch formulas, shift/affine variants match torch within float32 rounding; dtlz: evaluate parity ≤ few float32 ulps, pf parity ≤ 2e-6 (torch); cec2022: all 12 functions × dims {2,10,20} within rel 1e-5 vs torch. Unit tests: sibling `../../../../unit_test/etl/problems/` (parity tests may import torch).
