# evox_etl/problems — benchmark problems in functional style

## Intent
Port of the numerical problems from torch evox (`../../../evox/problems/`, read-only):
basic.py (Sphere, Rastrigin, Rosenbrock, Ackley, Griewank, Schwefel, Zakharov,
Levy...), dtlz.py (DTLZ1-7 + `pf` helpers), cec2022.py (CEC2022 f1-f12). Plain
functions in evoxc style — no classes, no OOP; the workflow traces them (per
`../DESIGN.md` §4.3, functions are NOT `@evoxc.defn`-wrapped).

Beyond numerical, the neuroevolution virtual Gaussian-noise ES problem is ported
in `neuroevolution/` (unifying the torch `VirtualProblem`/`VirtualLoRAProblem`
behind `lora_rank`; details in `neuroevolution/CONTEXT.md`), and the HPO wrapper
is ported as a HOST-SIDE redesign in `hpo_wrapper.py` (see below). NOT ported
(external-library dependent — reported to root): the remaining neuroevolution
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

## Release blocker — CEC2022 input data
`numerical/cec2022.py:27` computes `DATA_DIR = Path(__file__).resolve().parents[4] / "src" / "evox" / "problems" / "numerical" / "cec2022_input_data"`, i.e. the 54 `.txt` files are read out of the torch tree that is scheduled for deletion.
Deleting `src/evox/` makes every `CEC2022` trace raise `FileNotFoundError` (`cec2022.py:35`) in source checkouts; the expression is already wrong for an installed wheel (`parents[4]` = `site-packages`' parent).
No data ships with the port: `src/evox_etl/problems/numerical/` has no `cec2022_input_data/`, `MANIFEST.in` lists only the torch path, and `pyproject.toml` has no `package-data`/`[tool.setuptools]` section.
Fix shape (owner: this node + packaging owner): copy the `.txt` files under `src/evox_etl/problems/numerical/cec2022_input_data/`, load via `importlib.resources` (DESIGN.md §5.9), and add a manifest/package-data entry.
numpy stays allowed ONLY for loading that data; the values are baked as constant tensors.

## HPO parity vs torch (`hpo_wrapper.py`)
torch `HPOProblemWrapper(iterations, num_instances, workflow, num_repeats, copy_init_state)` IS an `evox.core.Problem`: it stacks the workflow's `Parameter`s (`torch.func.stack_module_state` + `vmap` + a `torch.library` custom-op loop keyed by `HPOData`/`__hpo_data__`) so an OUTER `StdWorkflow` optimizes the hyperparameters in-graph.
torch therefore tunes any `Parameter` of the workflow (arbitrary shape/tensor-valued, e.g. a `(7, 2)` matrix), evaluates `num_instances` candidates in parallel, and `copy_init_state` controls init-state reuse.
The port tunes only scalar fields of the inner algorithm/problem/monitor config dataclasses (`HPSlot(config_kind, field, transform)`), one candidate at a time, `num_repeats` = independent seeds, and the outer search must be host Python (`random_search` is the only driver shipped).
Consequence: no `evoxc` algorithm (e.g. PSO) can be the outer optimizer — the torch tutorials' flagship "outer StdWorkflow over the HPO wrapper" use case is not expressible, and there is no tensor/matrix-valued hyperparameter support.
Port-only API (no torch equivalent): `HPOProblemConfig` frozen config, `HPOFitnessMonitorConfig` (inner `EvalMonitorConfig` + `fit_aggregation` + host-side numpy `multi_obj_metric`), `HPSlot.transform`, `evaluate_many`, `random_search`.
torch API with no port equivalent: `HPOData`, `get_sub_state`, `get_init_params()` returning tensor `Parameter`s (port returns a `{dotted name: float}` dict), and `HPOFitnessMonitor.pre_tell`/`tell_fitness()` tensor protocol (port: `tell_fitness(workflow) -> float`).
Both keep `get_params_keys()`; both need `multi_obj_metric` for MO inners (port additionally needs `full_fit_history=True` on the inner monitor for `get_pf_fitness`).

## Migration surface for deleted torch neighbours
`examples/hpo/{single_objective_hpo,multi_objective_hpo,repeated_hpo}.py`, `docs/source/tutorial/tutorial_part5.md` and `docs/source/examples/{hpo,custom_hpo_prob}.ipynb` all import `evox.problems.hpo_wrapper` / `evox.*` and use the torch API, so they stop running when `src/evox/` is deleted.
`unit_test/problems/test_hpo_wrapper.py` (22 tests incl. outer-workflow MO HPO) is the torch suite; the port's equivalent is `../../../unit_test/etl/problems/test_hpo_wrapper.py` (10 host-side tests, no outer-workflow case).

## Notes for Agents
- cec2022 input data currently resolves into `../../../evox/problems/numerical/cec2022_input_data/` (see release blocker above).
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
| HPO wrapper (host-side) | `hpo_wrapper.py` | `HPOProblemWrapper` (NOT a `Problem`) + `random_search`; plain host Python, no traced nested workflow |
| Tests (canonical) | `../../../unit_test/etl/problems/` | sibling — numerical + virtual-problem tests; write access requires parent |
| Reference (torch) impl | `../../../evox/problems/` | sibling — READ-ONLY, never modify |
