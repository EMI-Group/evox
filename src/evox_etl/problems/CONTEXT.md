# evox_etl/problems — benchmark problems in functional style

## Intent
The numerical problems of the 2.0 `evox_etl` package (the 1.x torch implementation is
gone from the source tree, so nothing here may depend on it):
basic.py (Sphere, Rastrigin, Rosenbrock, Ackley, Griewank, Schwefel, Zakharov,
Levy...), dtlz.py (DTLZ1-7 + `pf` helpers), cec2022.py (CEC2022 f1-f12). Plain
functions in evoxc style — no classes, no OOP; the workflow traces them (per
`../DESIGN.md` §4.3, functions are NOT `@evoxc.defn`-wrapped).

Beyond numerical, the neuroevolution virtual Gaussian-noise ES problem lives in
`neuroevolution/` (the 1.x `VirtualProblem`/`VirtualLoRAProblem` unified
behind `lora_rank`; details in `neuroevolution/CONTEXT.md`), and the HPO wrapper
is a HOST-SIDE redesign in `hpo_wrapper.py` (see below). Not implemented
(external-library dependent — reported to root): the 1.x neuroevolution
problems (brax, mujoco_playground, supervised_learning).

## API Surface
- **Uniform problem signature**: `evaluate(config, problem_state, pop) ->
  (fitness, problem_state)` — plain function, `pop` shape `(n, dim)`, fitness
  `(n,)` (SO) or `(n, n_obj)` (MO). All problems minimize internally; the
  workflow applies `opt_direction` for maximization.
- **Configs** are frozen dataclasses mirroring the torch class `__init__`
  signatures; boundary/shift/affine tensor fields become numpy arrays in the
  config (baked once as graph constants inside functions, per DESIGN §4.1).
- **Statelessness**: numerical problems and the `neuroevolution` virtual problem
  are stateless and all share the empty frozen `ProblemState` (defined in
  `numerical/state.py`, re-exported at package level). The
  `(fitness, problem_state)` signature is kept for uniformity with stateful
  problems.
- **DTLZ reference fronts**: DTLZ1-7 expose `pf(config)` functions (plain
  functions meant to be traced) used by the metric/problem parity tests. Sampling
  comes from the canonical `evox_etl.operators.sampling` operators
  (`uniform_sampling`/
  `grid_sampling`, which return `(points, n_samples)` tuples — dtlz unpacks
  `[0]`). Verified bit-identical to the former private mirrors.
- **No boundary handling in basic problems**: raw math only, exactly like torch
  evox (constraining is the algorithm's responsibility there).
- Package `__init__` re-exports the torch `evox.problems.numerical` export
  surface plus `Zakharov`, `Levy`, `zakharov_func`, `levy_func` (torch basic.py
  has them but doesn't re-export) and `ProblemState`; it also re-exports the
  `neuroevolution` subpackage and its `VirtualProblem`/`VirtualLoRAProblem`
  aliases (the generic `evaluate`/`make_virtual_problem` are intentionally NOT
  hoisted to package level — name clash with the numerical `evaluate`).
- **HPO wrapper** (`hpo_wrapper.py`) is a HOST-SIDE functional HPO redesign
  (`HPOProblemWrapper`, `HPOProblemConfig`, `HPOFitnessMonitor`/`HPOMonitor`,
  `HPSlot`, `random_search`) — NOT an `evox_etl` `Problem`, so it cannot be the
  outer problem of a `StdWorkflow`; it builds/runs a fresh inner `StdWorkflow`
  per candidate in plain host Python (EvoX Compiler has no eager mode and cannot nest
  `evoxc.run` in a trace). See the module docstring for the precise limitations.

## CEC2022 input data (resolved)
The 53 `.txt` files ship WITH this package in `numerical/cec2022_input_data/`.
`numerical/cec2022.py` resolves `DATA_DIR = importlib.resources.files("evox_etl.problems.numerical").joinpath("cec2022_input_data")` — package-relative (DESIGN.md §5.9), so it works from a source checkout AND from an installed wheel, with no `src/evox/` path and no CWD dependence.
Packaging ships the data: `MANIFEST.in` includes `src/evox_etl/problems/numerical/cec2022_input_data/*.txt` and `pyproject.toml` has the matching `[tool.setuptools.package-data]` entry.
`importlib.resources.files(...).is_file()` guards the read, so a genuinely missing file still raises `FileNotFoundError` with the resolved path.
numpy stays allowed ONLY for loading that data; the values are baked as constant tensors.

## HPO parity vs the 1.x torch wrapper (`hpo_wrapper.py`)
torch `HPOProblemWrapper(iterations, num_instances, workflow, num_repeats, copy_init_state)` IS an `evox.core.Problem`: it stacks the workflow's `Parameter`s (`torch.func.stack_module_state` + `vmap` + a `torch.library` custom-op loop keyed by `HPOData`/`__hpo_data__`) so an OUTER `StdWorkflow` optimizes the hyperparameters in-graph.
torch therefore tunes any `Parameter` of the workflow (arbitrary shape/tensor-valued, e.g. a `(7, 2)` matrix), evaluates `num_instances` candidates in parallel, and `copy_init_state` controls init-state reuse.
The port tunes only scalar fields of the inner algorithm/problem/monitor config dataclasses (`HPSlot(config_kind, field, transform)`), one candidate at a time, `num_repeats` = independent seeds, and the outer search must be host Python (`random_search` is the only driver shipped).
Consequence: no `evoxc` algorithm (e.g. PSO) can be the outer optimizer — the torch tutorials' flagship "outer StdWorkflow over the HPO wrapper" use case is not expressible, and there is no tensor/matrix-valued hyperparameter support.
Port-only API (no torch equivalent): `HPOProblemConfig` frozen config, `HPOFitnessMonitorConfig` (inner `EvalMonitorConfig` + `fit_aggregation` + host-side numpy `multi_obj_metric`), `HPSlot.transform`, `evaluate_many`, `random_search`.
torch API with no port equivalent: `HPOData`, `get_sub_state`, `get_init_params()` returning tensor `Parameter`s (port returns a `{dotted name: float}` dict), and `HPOFitnessMonitor.pre_tell`/`tell_fitness()` tensor protocol (port: `tell_fitness(workflow) -> float`).
Both keep `get_params_keys()`; both need `multi_obj_metric` for MO inners (port additionally needs `full_fit_history=True` on the inner monitor for `get_pf_fitness`).

## Migration surface (1.x torch neighbours)
`examples/hpo/{single_objective_hpo,multi_objective_hpo,repeated_hpo}.py` and the `docs/` HPO material now import `evox_etl.*` and no longer use the torch API.
The port's HPO suite is the sibling `../../../unit_test/etl/problems/test_hpo_wrapper.py` (host-side only, no outer-workflow case).
The 1.x torch suites (`unit_test/problems/`, etc.) are outside this node and go away with the torch tree.

## Notes for Agents
- CEC2022 input data resolves package-relatively (see "CEC2022 input data (resolved)" above) — never reintroduce a `src/evox/...` path.
- **Test suite** (numerical + neuroevolution virtual-problem; pure-evoxc + torch
  parity): canonical home is the sibling `../../../unit_test/etl/problems/`
  (write access requires parent). Gate: `<venv>/bin/python -m pytest
  unit_test/etl/problems -q`.

## EvoX Compiler issues found (escalated to root — also needs adding to ../CONTEXT.md)
Hit while porting the numerical problems; all worked around in this node (see
`numerical/CONTEXT.md` "Verified evoxc facts" for the workaround patterns):
1. **Slice op can't express full-axis `:` over a dynamic dim** — `x[:, a:b]` /
   `x[:, :]` fail at trace/lower when the batch dim is symbolic. Repro:
   `evoxc.build(lambda x: x[:, 1:], TensorSpec((None, 4), np.float32))` → lower
   error. Workaround: `evoxc.gather(x, const_int32_arange, axis=1)`.
2. **Reductions directly over a trace-input leaf fail** — `enp.sum(x, axis=1)`
   where x is the function's input leaf raises a shape mismatch (IR Dim-vs-None
   inconsistency). Repro: `evoxc.build(lambda x: evoxc.sum(x, axis=1),
   TensorSpec((2, 3), np.float32))`. Workaround: apply an elementwise op first
   (`x * 1.0`).
3. **`enp.expand_dims`/`enp.reshape` can't carry dynamic (None) dims** — they
   reject None entries and Dim entries fail at lowering. Workaround: unroll
   static Python loops (see `katsuura_func` in cec2022.py).
4. **`!=` is not overloaded on symbolic tensors** (`x != 0` raises TraceError;
   only `== < > <= >=` work). Use `evoxc.not_equal`.
5. **No `*_like` ops** (`zeros_like`/`ones_like`/`full_like` absent in
   evoxc.numpy) — zero-init via `x * 0.0`.
6. **`enp.floor` missing** (top-level `evoxc.floor` exists); similarly top-level
   `evoxc.zeros`/`evoxc.ones` are CONCRETE creators and must not be used inside
   traces (`enp.zeros`/`enp.ones` are the traced versions) — easy to mix up.
7. **`float32 ** int64` promotes to float64** — cast exponents to float32
   explicitly.
8. **Numpy arrays inside config dataclasses are legal static trace leaves on
   the installed evoxc master** (ndarray accepted by `evoxc.build`) — basic.py
   configs use no pytree registration: as plain-leaf pytrees they keep evoxc's
   by-value static revalidation, so a drifted config at `evoxc.run` raises
   TraceError. See `numerical/CONTEXT.md` "Config design (current state)".
   The `make_*` constructor policy does not apply to numerical problems
   (DESIGN.md §4.1 carve-out: they keep their validation-only
   `__post_init__`s).
9. **`evoxc.gather` + `enp.expand_dims` interaction is confusing** (see
   cec2022.py comments); `evoxc.gather` accepts int32 indices, np.take semantics.
10. **`TensorSpec` API quirks**: no `from_tensor` method; shape must be a flat
    tuple; built `Executable` has no `.run` method (use top-level
    `evoxc.run(exe, *args)` with ALL positional args, static ones included).
    `evoxc.evaluate` rejects static args entirely (configs must go through
    `evoxc.build`).
11. **evoxc operators reject numpy scalars as operands** — `x / np.float32(2.0)`
    raises `TypeError: ... numpy scalars/arrays are not Python scalars` (evoxc
    only accepts SymbolicTensor or Python `bool`/`int`/`float`/`complex` as
    operands). Repro: `evoxc.build(lambda x: x / np.float32(2.0),
    TensorSpec((4,), np.float32))`. Workaround: pass a Python float (`x / 2.0`).

## Routing Table
| Area | Path | Notes |
|---|---|---|
| Numerical problems (basic, dtlz, cec2022, state) | `numerical/` | single subpackage; owns `ProblemState` |
| Neuroevolution (virtual Gaussian-noise problem) | `neuroevolution/` | payload protocol `(center_flat, seeds, sigma)`; shared `ProblemState`; see `neuroevolution/CONTEXT.md` |
| HPO wrapper (host-side) | `hpo_wrapper.py` | `HPOProblemWrapper` (NOT a `Problem`, scalar config-field slots only) + `random_search`; plain host Python, no traced nested workflow |
| Tests (canonical) | `../../../unit_test/etl/problems/` | sibling — numerical, virtual-problem + host-side HPO tests; write access requires parent |
