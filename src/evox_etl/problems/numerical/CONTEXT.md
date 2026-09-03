# evox_etl/problems/numerical — benchmark problems in functional style

## Intent
Functional-ETL port of the torch evox numerical problems (`../../../evox/problems/numerical/`, read-only reference), math mirrored 1:1. Each problem is a frozen config dataclass + plain module-level `evaluate(config, problem_state, pop) -> (fitness, problem_state)` (PLAIN functions, NO `@etl.defn` — DESIGN §4.3). DTLZ adds `pf(config)` reference-Pareto-front functions. Numerical problems are stateless: they share the empty frozen `ProblemState` (`state.py`); the uniform signature is kept anyway.

NOT ported (external-library dependent — reported to root): neuroevolution problems (brax, mujoco_playground, supervised_learning, virtual_lora), hpo_wrapper.

## API Surface
- `basic.py`: 10 configs (`ShiftAffineNumericalProblem`, `Ackley(a=20.0, b=0.2, c=2π)`, `Griewank`, `Rastrigin`, `Rosenbrock`, `Schwefel`, `Sphere`, `Ellipsoid`, `Zakharov`, `Levy` — shift/affine numpy-array fields) + 9 `*_func`s (torch signatures) + module-level `evaluate` dispatching statically on config type. NO boundary handling (raw math — matches torch).
- `dtlz.py`: `DTLZ1(d=7)/DTLZ2..6(d=12)/DTLZ7(d=21)` configs (m=3, ref_num=1000) + `evaluate`/`pf` dispatchers → per-variant `evaluate_dtlzN`/`pf_dtlzN` (DTLZ3/DTLZ4 share `pf_dtlz2`, mirroring torch inheritance) + private `_uniform_sampling` (Das-Dennis) / `_grid_sampling` mirroring torch `operators/sampling/{uniform,gird}.py` (operators milestone must dedupe them).
- `cec2022.py`: `CEC2022(problem_number, dimension)` config + `evaluate` + all internal math as plain functions (`cec2022_f1..f12`, `shift/rotate/cut/sr_func_rate/cf_cal`, basic funcs). Imports 5 basic funcs from `basic`.
- `__init__.py`: re-exports all classes, funcs, `CEC2022`, `DTLZ1-7`, `ProblemState`, and the three modules.

## Verified etl facts (do not re-investigate)
- `etl.evaluate` REJECTS static (non-tensor) args. Use `etl.build(fn, cfg, state, spec)` + `etl.run(exe, cfg, state, x)` — run() wants ALL positional args (static ones by value); `Executable` has no `.run` method; `TensorSpec` shape is a flat tuple `(n, d)` (no `from_tensor`).
- Config dataclasses holding numpy arrays (basic.py shift/affine) are registered as zero-child pytree nodes (`etl.register_pytree_node`) so `etl.build` passes them through as opaque static data. Plain int-field dataclasses (dtlz/cec2022) work as static args directly.
- `x[:, a:b]` slices FAIL whenever an axis keeps a full-axis `:` over a dynamic dim → use `etl.gather(x, const_int32_arange, axis=1)` (np.take semantics, result `(n, len(idx))`). Static int column indexing (`x[:, 0]`) works. DTLZ formulas slice heavily → traced shapes must be fully STATIC (fine: workflow builds specs from concrete tensors; tests must use static-shape specs).
- Reductions directly over a trace-input leaf fail (IR Dim-vs-None inconsistency) → apply an elementwise op first (`x * 1.0`); no `zeros_like`/`full_like` in enp → zero-init via `x * 0.0`.
- `enp.expand_dims`/`reshape` cannot carry dynamic dims → unroll static Python loops where needed (see `katsuura_func`).
- `!=` not overloaded on symbolic tensors → `etl.not_equal`. `enp.floor` missing → top-level `etl.floor`. Top-level `etl.zeros/ones` are CONCRETE creators → use `enp.zeros/ones` inside traces. `float32 ** int64` promotes to float64 → cast exponents to float32. `etl.dot` needs rank ≥ 2 → `etl.matmul` (numpy semantics). `etl.gather` accepts int32 indices.
- Constants: `etl.constant(etl.tensor(np.asarray(..., dtype=np.float32)))` baked INSIDE evaluate (never closure-captured concrete tensors). When a shift is None, add a zero constant instead (static Python branch).

## Data location
cec2022 input data lives in `../../../evox/problems/numerical/cec2022_input_data/` (torch source tree); loaded as numpy with `DATA_DIR = Path(__file__).resolve().parents[4] / "src" / "evox" / "problems" / "numerical" / "cec2022_input_data"` (memoized module-level cache). numpy allowed ONLY for that loading; never in graph code.

## Validation
Pattern: `etl.build` + `etl.run` (numpy backend) vs references — basic: bit-exact vs numpy versions of the torch formulas, shift/affine variants match torch within float32 rounding; dtlz: evaluate parity ≤ few float32 ulps, pf parity ≤ 2e-6 (torch); cec2022: all 12 functions × dims {2,10,20} within rel 1e-5 vs torch. Unit tests: sibling `../../../../unit_test/etl/problems/` (parity tests may import torch).
