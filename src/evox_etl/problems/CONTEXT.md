# evox_etl/problems — benchmark problems in functional style

## Intent
Port of the numerical problems from torch evox (`../../../evox/problems/`, read-only):
basic.py (Sphere, Rastrigin, Rosenbrock, Ackley, Griewank, Schwefel, Zakharov,
Levy...), dtlz.py (DTLZ1-7 + helpers), cec2022.py (with shipped input data). Plain
functions in etl style — no classes, no OOP; the workflow traces them (per
`../DESIGN.md` §4.3, functions are NOT `@etl.defn`-wrapped).

NOT ported (external-library dependent — reported to root): neuroevolution problems
(brax, mujoco_playground, supervised_learning, virtual_lora), hpo_wrapper.

## API Surface
- **Uniform problem signature**: `evaluate(config, problem_state, pop) ->
  (fitness, problem_state)` — plain function, `pop` shape `(n, dim)`, fitness
  `(n,)` (SO) or `(n, n_obj)` (MO). All problems minimize internally; the
  workflow applies `opt_direction` for maximization.
- **Configs** are frozen dataclasses mirroring the torch class `__init__`
  signatures; boundary/shift/affine tensor fields become numpy arrays in the
  config (baked once as graph constants inside functions, per DESIGN §4.1).
- **Statelessness**: numerical problems are stateless and all share the empty
  frozen `ProblemState` (defined in `numerical/state.py`, re-exported at
  package level). The `(fitness, problem_state)` signature is kept for
  uniformity with stateful problems.
- **DTLZ reference fronts**: DTLZ1-7 expose `pf(config)` functions (plain
  functions meant to be traced) used by metrics/tests. Sampling comes from the
  canonical `evox_etl.operators.sampling` operators (`uniform_sampling`/
  `grid_sampling`, which return `(points, n_samples)` tuples — dtlz unpacks
  `[0]`). Verified bit-identical to the former private mirrors.
- **No boundary handling in basic problems**: raw math only, exactly like torch
  evox (constraining is the algorithm's responsibility there).
- Package `__init__` re-exports the torch `evox.problems.numerical` export
  surface plus `Zakharov`, `Levy`, `zakharov_func`, `levy_func` (torch basic.py
  has them but doesn't re-export) and `ProblemState`.

## Notes for Agents
- cec2022 input data lives in `../../../evox/problems/numerical/cec2022_input_data/`
  — numpy allowed ONLY for loading that data; bake it as constant tensors (closure).
- **Test suite** (131 tests, GREEN on the numpy backend) lives at `tests/` inside
  this node (pure-etl + torch parity under `tests/parity/`). Canonical home is
  the sibling `../../../unit_test/etl/problems/` — this node has no write access
  to siblings, so the parent must relocate the files (they are relocate-ready:
  self-contained, conftest shim inserts repo root + src into sys.path) and run
  the gate `/mnt/local-ssd/bchuang/evox/.venv/bin/python -m pytest
  unit_test/etl/problems -q`. Run locally with `... -m pytest
  src/evox_etl/problems/tests -q`.
- **.gitignore gotcha**: the repo-root `.gitignore` has a bare `tests` entry
  that matches `src/evox_etl/problems/tests/` — new files there must be added
  with `git add -f` until the parent adds a negation pattern (the suite files
  are already force-added/tracked).

## ETL issues found (escalated to root — also needs adding to ../CONTEXT.md)
Hit while porting the numerical problems; all worked around in this node (see
`numerical/CONTEXT.md` "Verified etl facts" for the workaround patterns):
1. **Slice op can't express full-axis `:` over a dynamic dim** — `x[:, a:b]` /
   `x[:, :]` fail at trace/lower when the batch dim is symbolic. Repro:
   `etl.build(lambda x: x[:, 1:], TensorSpec((None, 4), np.float32))` → lower
   error. Workaround: `etl.gather(x, const_int32_arange, axis=1)`.
2. **Reductions directly over a trace-input leaf fail** — `enp.sum(x, axis=1)`
   where x is the function's input leaf raises a shape mismatch (IR Dim-vs-None
   inconsistency). Repro: `etl.build(lambda x: etl.sum(x, axis=1),
   TensorSpec((2, 3), np.float32))`. Workaround: apply an elementwise op first
   (`x * 1.0`).
3. **`enp.expand_dims`/`enp.reshape` can't carry dynamic (None) dims** — they
   reject None entries and Dim entries fail at lowering. Workaround: unroll
   static Python loops (see `katsuura_func` in cec2022.py).
4. **`!=` is not overloaded on symbolic tensors** (`x != 0` raises TraceError;
   only `== < > <= >=` work). Use `etl.not_equal`.
5. **No `*_like` ops** (`zeros_like`/`ones_like`/`full_like` absent in
   etl.numpy) — zero-init via `x * 0.0`.
6. **`enp.floor` missing** (top-level `etl.floor` exists); similarly top-level
   `etl.zeros`/`etl.ones` are CONCRETE creators and must not be used inside
   traces (`enp.zeros`/`enp.ones` are the traced versions) — easy to mix up.
7. **`float32 ** int64` promotes to float64** — cast exponents to float32
   explicitly.
8. **Numpy arrays inside config dataclasses are legal static trace leaves on
   the installed etl master** (ndarray accepted by `etl.build`) — basic.py
   configs use no pytree registration: as plain-leaf pytrees they keep etl's
   by-value static revalidation, so a drifted config at `etl.run` raises
   TraceError. See `numerical/CONTEXT.md` "Config design (current state)".
   The `make_*` constructor policy does not apply to numerical problems
   (DESIGN.md §4.1 carve-out: they keep their validation-only
   `__post_init__`s).
9. **`etl.gather` + `enp.expand_dims` interaction is confusing** (see
   cec2022.py comments); `etl.gather` accepts int32 indices, np.take semantics.
10. **`TensorSpec` API quirks**: no `from_tensor` method; shape must be a flat
    tuple; built `Executable` has no `.run` method (use top-level
    `etl.run(exe, *args)` with ALL positional args, static ones included).
    `etl.evaluate` rejects static args entirely (configs must go through
    `etl.build`).

## Routing Table
| Area | Path | Notes |
|---|---|---|
| Numerical problems (basic, dtlz, cec2022, state) | `numerical/` | single subpackage; owns `ProblemState` |
| Test suite (relocate-ready) | `tests/` | in-node copy; canonical home is the sibling unit_test dir — parent relocates |
| Tests (canonical) | `../../../unit_test/etl/problems/` | sibling — write access requires parent |
| Reference (torch) impl | `../../../evox/problems/` | sibling — READ-ONLY, never modify |
