# evox_etl/problems/numerical — benchmark problems in functional style

## Intent
Functional ETL ports of the torch numerical problems
(`../../../evox/problems/numerical/`, read-only reference), math mirrored
1:1. Each problem = frozen config dataclass + plain `evaluate`/`pf`
functions (NO `@etl.defn` — plain callables only, verified contract).

## dtlz.py — DTLZ1-7 (DONE)
- Configs: frozen dataclasses mirroring torch `__init__` defaults, no
  `device` field. Torch's `DTLZ` base class disappears; `_sample` helper
  holds its uniform sampling.
- `evaluate(config, problem_state, X) -> (f, problem_state)` and
  `pf(config)` are module-level **dispatchers** keyed on config type
  (static `isinstance` at trace time — the config is a compile-time static
  arg; one module holds 7 variants so name collisions are impossible).
  Per-variant implementations: `evaluate_dtlzN` / `pf_dtlzN`; DTLZ3/DTLZ4
  share `pf_dtlz2` (mirrors torch inheritance). **Sibling modules with
  multiple problems per file should use the same dispatch pattern.**
- Private `_uniform_sampling` (Das-Dennis) / `_grid_sampling` mirror torch
  `operators/sampling/{uniform,gird}.py` math exactly — the operators
  milestone must move/dedupe them.
- **Static-shape requirement (important):** etl's IR slice op cannot
  express full-axis slices over symbolic dims, and DTLZ formulas slice
  `X[:, m-1:]` etc. → traced shapes must be fully STATIC (batch `n` AND
  dim `d`). StdWorkflow builds specs from concrete tensors, so fine in
  practice; tests must use static-shape `TensorSpec((n, d), ...)`.
- Validated vs torch (throwaway script, not committed): evaluate parity
  ≤ few float32 ulps; pf parity ≤ 2e-6 (DTLZ7 pf ~1.5e-8); uniform
  sampling bit-exact; grid sampling within 1 ulp (etl.linspace backend
  kernels differ from torch's linspace kernel by ~1 ulp — irrelevant at
  the 1e-4 pf tolerance).

## Verified run pattern (numpy backend)
```python
exe = etl.build(fn, cfg, ProblemState(), TensorSpec((n, d), np.dtype("float32")))
f, ps = etl.run(exe, cfg, ProblemState(), x)   # run() wants ALL positional
                                               # args, static ones by value
exe = etl.build(pf, cfg)                       # pf: no tensor inputs
front = etl.run(exe, cfg)
```
(`Executable` has no `.run` method — use top-level `etl.run`. `etl.bind`
binds only NAMED tensor inputs. Python scalars in expressions follow
numpy weak-scalar rules → float32 stays float32; `float32 ** int64`
promotes to float64 — cast exponents to float32. `etl.zeros/ones/linspace`
top-level are CONCRETE creators — inside traces use `enp.zeros/ones` and
`etl.linspace` with explicit `dtype=etl.float32`.)

## Not ported (external-library dependent — reported to root)
neuroevolution problems (brax, mujoco_playground, supervised_learning,
virtual_lora), hpo_wrapper.
