# evox — Functional EvoX on EvoX Compiler

## Intent
The EvoX framework on the EvoX Compiler tensor library
(`evoxc`, declared as a runtime dependency in the root `pyproject.toml` and
installed into a local venv — see "Environment" below). This package IS the shipped 2.0 release package
(distribution `evox`, version `2.0.0a1`, single-sourced in the root
`pyproject.toml`): the 1.x torch OOP implementation is gone from the source tree,
so no module here may import or depend on it.
The functional design replaces it: frozen config dataclasses, namedtuple/dataclass tensor states, and plain module-level
functions (no `@evoxc.defn` — see DESIGN.md §4.3) in the step protocol
(`init(config, key) -> state`, `step(config, state, evaluate) -> state`), and a
compile-once StdWorkflow loop.
`evox.__version__` is read from the installed distribution metadata and
falls back to `"0.0.0.dev0"` in a source checkout.
**Binding spec: `DESIGN.md` in this directory — read it fully before writing any code.**

## API Surface
- `evox.core`: `Algorithm`/`Problem`/`Monitor` protocols (duck-typed), `StdWorkflow`,
  `WorkflowState`, state helpers.
- `evox.algorithms`: SO (de/es/pso variants, incl. the training-free `virtual_es`/
  `virtual_lora_es` family) + MO (nsga2, nsga3, moead, rvea,
  rveaa, hype) — each module = config dataclass + `make_*` constructor +
  `init`/`init_step`/`step` plain functions (step protocol, see
  `core/algorithm.py`). Public surface keeps the 1.x names: `VirtualES`/`VirtualLoRAES`
  are the bare names (both alias the VirtualES config), the distinct low-rank
  config is `VirtualLoRAESConfig`. The virtual-population ES family adds
  `virtual_es.py` (`VirtualES`/`VirtualLoRAES` + `make_virtual_es`),
  `virtual_lora_es.py` (`VirtualLoRAESConfig` + `make_virtual_lora_es`) and the
  shared deterministic-noise module `virtual_noise.py`.
  The bare 1.x algorithm names are importable at every level: `evox.algorithms`,
  `.algorithms.so`, `.algorithms.mo` (`NSGA2`/`NSGA3`/`RVEA`/`RVEAa`/`MOEAD`/`HypE`)
  and `.algorithms.so.es_variants` (`OpenES`, `XNES`, `SeparableNES`, `SNES`, `ARS`,
  `ASEBO`, `DES`, `ESMC`, `GuidedES`, `NoiseReuseES`, `PersistentES`, `CMAES`,
  `VirtualES`, `VirtualLoRAES`) — each is an alias of the matching `*Config`
  dataclass, next to the native `*Config`/`make_*` spelling.
- `evox.operators`: pure functions (sampling, selection, crossover, mutation).
- `evox.problems.numerical`: basic, dtlz, cec2022.
- `evox.problems.neuroevolution`: `VirtualProblem`/`VirtualLoRAProblem`
  (unified behind `lora_rank`) + `make_virtual_problem` + `evaluate`
  (payload protocol `(center_flat, seeds, sigma)`).
- `evox.problems.hpo_wrapper`: HOST-side HPO (`HPOProblemWrapper`, `HPSlot`,
  HPO monitor configs, `random_search`) — re-exported from `evox.problems`
  (NOT an `evox` `Problem`).
- `evox.metrics`: igd, gd, hv. `evox.workflows`: std_workflow, eval_monitor.
- `evox.utils`: tree helpers, min_by, dominate_relation, pairwise distances,
  parse_opt_direction, rank — plus re-exports of the 1.x `evox.utils` jit-fix helpers
  (`switch`, `clamp`/`clamp_int`/`clamp_float`, `clip`, `maximum*`/`minimum*`,
  `lexsort`, `nanmin`, `nanmax`, `randint`), whose canonical home stays
  `evox.operators.jit_fix_operator`.

## Constraints
- NO eager tensor ops — everything inside `@evoxc.defn`/traces (EvoX Compiler has no eager mode).
- Config fields are plain static leaves (Python scalars + flat float tuples)
  partialized away at compile time; state leaves are tensors ONLY. Minimization
  semantics internally (workflow applies opt_direction).
- No torch/numpy in framework code (numpy allowed only to load cec2022 input data and
  for host-side make_* preprocessing).
- EvoX Compiler issues found while implementing: record under "EvoX Compiler issues
  found" below and escalate to the root agent (never patch the `evoxc` package from
  here).

## Routing Table
| Area | Path | Notes |
|---|---|---|
| Core (protocols, workflow, monitor, state) | `core/` | Foundation — implement FIRST |
| Operators (pure functions) | `operators/` | sampling/selection/crossover/mutation |
| Algorithms SO | `algorithms/so/` | de_variants, es_variants (incl. `virtual_es.py`/`virtual_lora_es.py`/`virtual_noise.py`), pso_variants |
| Algorithms MO | `algorithms/mo/` | nsga2, nsga3, moead, rvea, rveaa, hype |
| Config helpers (shared) | `algorithms/_config_utils.py` | make_* support: to_float_tuple, normalize_bounds, require_*, bake_* |
| Numerical problems | `problems/numerical/` | basic, dtlz, cec2022 |
| Neuroevolution problems | `problems/neuroevolution/` | VirtualProblem/VirtualLoRAProblem (virtual Gaussian-noise payload); see `problems/neuroevolution/CONTEXT.md` |
| HPO wrapper (host-side) | `problems/hpo_wrapper.py` | HPOProblemWrapper/HPSlot/random_search (NOT a `Problem`); numpy host-side only |
| Metrics | `metrics/` | gd/gd_plus, igd/igd_plus, hv + MC variants; tests at `../unit_test/evox/metrics/` |
| Workflow + EvalMonitor | `workflows/` | std_workflow re-export, eval_monitor; tests at `../unit_test/evox/workflows/` |
| Visualization | `vis_tools/` | Plotly figure builders + EvoXVision `.exv` serialization; tests at `../unit_test/evox/vis_tools/` |
| Utilities | `utils/` | functional helpers |
| End-to-end tests (e2e + autoload) | `../unit_test/evox/` | real-algorithm convergence + autoload-guard tests (mirror) |
| Tests | `../unit_test/evox/` | sibling — mirrors this package |
| Benchmarks (backend comparison) | `../benchmarks/evox_vs_torch/` | sibling — frozen `evox-*` backend ids + committed result JSONs |
| EvoX Compiler (`evoxc`) | installed dependency | `evoxc >= 0.1.0`; fixes only by the root agent |

## Environment
Build a venv with `uv venv` + `uv pip install -e . pytest plotly pandas`; `evoxc`
(`>= 0.1.0`) and `numpy` come from `pyproject.toml`, and a `evoxc` source checkout can
be installed by path instead of by name. Canonical suite: the sibling
`../unit_test/evox/` (mirrors this package) —
`PYTHONPATH=src <venv>/bin/python -m pytest -q` from the repo root (numpy
backend; ~6 min; per-suite counts in `../unit_test/evox/CONTEXT.md`).
`evox` and `evox_ext` are PEP 420 namespace packages — drivers need
`PYTHONPATH=src`.
Lint: `ruff check src/evox src/evox_ext` (config in the root `pyproject.toml`);
the tree still carries pre-existing `I001`/`W292`/`F401` findings in
`algorithms/` — do not introduce new ones.

## EvoX Compiler issues found (escalated to root agent)
1. **np.ndarray fields in config dataclasses ARE accepted as static trace values
   by the installed `evoxc`** (master @f2f50a7, incl. commit b8062a9 "accept np.ndarray
   as static trace values" — empirically verified via runtime probes, numpy
   backend). Config construction follows DESIGN.md §4.1: algorithm configs are
   dumb frozen dataclasses storing plain static leaves (Python scalars + flat
   float tuples); array acceptance, normalization, and ValueError validation
   live in module-level `make_*` constructors (shared helpers in
   `algorithms/_config_utils.py`). Remaining constraints: (a) non-None
   callable fields are STILL rejected as static leaves anywhere in the pytree, so
   zero-child registration is reserved for callable-bearing configs only
   (mo/nsga3, moead, rvea — keep it there); (b) zero-child registration makes the
   config an opaque node — `evoxc.run` performs NO by-value static revalidation,
   while plain-leaf tuple/ndarray fields raise TraceError on a drifted value;
   (c) plain-float-tuple lb/ub storage is the package convention for frozen-
   dataclass `__eq__`/`__hash__` (frozen=True + ndarray is broken/unhashable).
   Numerical problems are the carve-out: basic.py/cec2022.py keep validation-only
   `__post_init__`s (AssertionError with message, frozen because `../unit_test/evox`
   asserts them — do NOT convert to ValueError).
2. **`evoxc.select` does not numpy-broadcast a `(n,)` condition against `(n, m)`
   branches** (`cannot broadcast incompatible dims n and m`). numpy broadcasts
   `(n,)` → `(n, 1)` against `(n, m)` fine. Workaround: always
   `enp.expand_dims(cond, 1)` first (scalar conds broadcast OK).
   Repro: `evoxc.build(lambda x: evoxc.select(evoxc.sum(x, axes=1) < 0.5, 0.0, x),
   TensorSpec((3, 4), float32))` → ShapeError.
3. **`SymbolicTensor.__getitem__` rejects `None`/newaxis indices** — use
   `enp.expand_dims`/`enp.reshape` (pso_variants).
4. **`evoxc.scatter` rejects scalar (rank-0) updates** despite the docstring —
   rank-matching workaround (pso_variants).
5. **`evoxc.dot` requires rank ≥ 2** — vector·matrix products need expand_dims
   tricks (es_variants; documented in its CONTEXT.md).
6. Minor es_variants discoveries (documented in its CONTEXT.md): `evoxc.transpose`
   axes must be a tuple; `evoxc.clamp` needs both bounds; `(k,n)*(k,)` does not
   align over k (use expand_dims); `evoxc.svd` returns reduced (U,S,Vh).
7. **No scatter-add** (`evoxc` `scatter` is replacement-only put_along_axis) —
   worked around with one-hot segment sums (mo/nsga3); a composition note, not a bug.
8. Cosmetic: the `evoxc` numpy backend leaks a `RuntimeWarning: invalid value
   encountered in divide` for the intentional inf-beta draws (SBX/SHADE/SaDE NaN
   paths) — deliberate, no change made.
### Additional findings (core/operators/problems teams)
9. **Inconsistent axis arg naming**: reductions take `axes=` but topk/argmin/gather/
   `enp.sum/min` take `axis=`. Check the op signature before using.
10. `evoxc.tree_unflatten(leaves, treespec)` argument order is REVERSED vs JAX.
11. Top-level `evoxc.zeros/ones/full/empty` are EAGER concrete creators (raise inside
    traces); in-graph constants: `evoxc.ops.constant(evoxc.core.tensor(np.asarray(...)))`
    — `constant()` rejects raw ndarray; `x * 0.0` for zero-init; no `*_like` ops;
    `enp.floor` missing.
12. Dtype promotion is stricter than numpy: int32+int→int64, int32*float→float64,
    float32**int64→float64 — explicit `evoxc.cast` needed.
13. `TensorSpec.from_tensor` does NOT exist (flat `(shape, dtype)` only);
    `Executable` has no `.run` (use `evoxc.run(exe, *args)`); `evoxc.evaluate` rejects
    static args (use `evoxc.build`); scalar tensor inputs must be 0-d ndarrays, not
    numpy scalars. Statics are path-specialized per distinct value — a different
    static value at run raises TraceError (build a new exe per value).
14. `evoxc.gather` is numpy-take semantics (it does not gather along an axis);
    take_along_axis needs a
    flatten trick; `!=` on symbolic tensors needs `evoxc.not_equal`; `evoxc.squeeze`
    missing at top level; slice op cannot express full-axis `:` over dynamic dims
    (`evoxc.gather(x, arange, axis=1)` workaround); reductions directly over a
    trace-input leaf can fail (multiply by 1.0 first); `enp.expand_dims/reshape`
    cannot carry dynamic dims (unroll static loops).
15. **`evoxc` xla adapter GPU client-exhaustion bug** (fresh PJRT client per compile/load
    → process SIGABRT; real jax_cuda12 plugin) — FIXED and merged into `evoxc` master
    (commit 838739c, shared refcounted client; the venv needs no task branch).
    xla-cuda runs still need cuDNN ≥9.8 via LD_LIBRARY_PATH (plugin compiled vs
    cuDNN 9.8.0; venv ships 9.1.0).
16. **`evoxc.run` returns concrete `evoxc.core.tensor.Tensor` objects, NOT ndarrays** —
    `np.asarray(result)` yields an object-dtype 0-d array; always use
    `result.numpy()`. Host-side code reading step results must call `.numpy()`
    (and `.to(evoxc.core.Device("cpu"))` first if running on a non-cpu device).
17. **Static args at `evoxc.build` vs `evoxc.run` must match exactly**: positional
    statics passed to `evoxc.build` must be RE-passed (by value) to `evoxc.run`;
    parameters omitted at build (Python defaults) are baked into the executable and
    must NOT be passed at run (else "run-time input structure does not match the
    traced signature" TraceError).
18. Positive finds (no workaround needed): `evoxc.random.uniform` accepts tensor
    low/high bounds; `evoxc.random.split_n(key, n)` returns a tuple of n keys;
    `evoxc.norm(x)` defaults to L2 over all axes (`axis=None, ord=2`) and takes the
    singular `axis=` kwarg like topk/argmin/gather (issue 9); `evoxc.gather` accepts
    0-d scalar indices and squeezes (the pso `(1,)`-reshape workaround is
    unnecessary); float32 ** Python-float exponent stays float32; `evoxc` has no
    any/all ops — compose via `evoxc.max`/`evoxc.min` over bool axes.
19. **`evoxc` operators REJECT numpy scalars/arrays as operands** — `x / np.float32(2.0)`
    raises `TypeError: numpy scalars/arrays are not Python scalars` (`evoxc` operands
    must be SymbolicTensor or Python `bool`/`int`/`float`/`complex`). Pass a plain
    Python float and never mix a numpy value with a symbolic tensor inside a trace
    (hit by the virtual ES + virtual-problem host constants).
20. Positive find (no workaround needed): batched 3-D `evoxc.dot((pop, d, r),
    (pop, r, k))` works — used by the LoRA-mode gradient in `virtual_lora_es.py`.

## Config construction (current state)
The `__post_init__`-removal / functional-constructor refactor is COMPLETE for
algorithms: every algorithm config in `algorithms/` is a dumb frozen dataclass
storing only plain static leaves (Python scalars + flat float tuples); a
module-level `make_<algorithm>` constructor in the SAME module as the config
(StdWorkflow resolves the module via `type(config).__module__`) does all
normalization (ndarray→flat tuple via `algorithms/_config_utils.py`), ValueError
validation (no bare asserts), and eager derived defaults (XNES/SeparableNES
pop_size/lr, ASEBO subspace_dims — `_discover_pop_size` reads `cfg.pop_size`).
The old read-only design audit (37 `__post_init__` defs, the zero-child
`register_pytree_node` hack, call-site inventories) is superseded — preserved in
git history. Current-state notes per subtree live in the child CONTEXT.md files
(algorithms/so/de_variants, es_variants, pso_variants, algorithms/mo,
problems/numerical); binding policy in DESIGN.md §4.1.
Exports: `make_*` at each family `__init__`, `algorithms/so/__init__.py`, and
`algorithms/__init__.py` (1.x-style bare-name aliases unchanged).
Carve-out: numerical problems (basic.py, cec2022.py) keep their validation-only
`__post_init__`s — direct dataclass construction is their sanctioned public API.
Unit-test construction sites at `../unit_test/evox` migrate to `make_*` in a
parallel wave (direct construction still works for pre-normalized values but
bypasses make_* validation).

## Public-surface deltas vs the 1.x line (removed in 2.0; matters for migration)
- Root `evox` re-exports core protocols, `StdWorkflow`, `EvalMonitor(Config)`,
  the 5 metric functions and `random`/`key`; it does NOT explicitly import
  `algorithms`/`problems`/`operators` — those become package attributes only because
  `auto_load_extensions()` imports the 5 domains as a side effect.
- `evox.utils` re-exports the 1.x `evox.utils` jit-fix helpers
  (`switch`/`clamp`/`clamp_int`/`clamp_float`/`clip`/`maximum*`/`minimum*`/`lexsort`/
  `nanmin`/`nanmax`/`randint`) alongside the native helpers, so those 1.x imports
  keep resolving; the canonical implementation home is `operators/jit_fix_operator.py`.
  `ParamsAndVector`, `VmapInfo`/`register_vmap_op` have no counterpart and are NOT
  re-exported (virtual problems take pre-flattened `param_shapes`/`layer_specs`).
- Bare 1.x algorithm names resolve at every algorithmic level: `evox.algorithms`,
  `.algorithms.so`, `.algorithms.mo` (aliases of the `*Config` dataclasses) and
  `.algorithms.so.es_variants` / `.de_variants` / `.pso_variants`.
- `metrics.hv` takes a leading RNG key (`hv(key, objs, ref, num_sample)`); the package
  adds `gd_plus`/`igd_plus`/`bounding_cube_monte_carlo_hv`/`each_cube_monte_carlo_hv`.
- DTLZ exposes the Pareto front as module-level `pf(config)`/`pf_dtlzN`, not a
  `DTLZ.pf()` method; there is no `DTLZ` base class.
- `vis_tools` exports all `plot_*` + `EvoXVisionAdapter`/`new_exv_metadata` at package
  level (the 1.x `vis_tools.__init__` was empty) and imports Plotly lazily.

## Packaging
The package ships as distribution `evox` version `2.0.0a1`, single-sourced in the root
`pyproject.toml`:
- CEC2022 input data ships WITH the package: the 53 `.txt` files live in
  `problems/numerical/cec2022_input_data/`, `numerical/cec2022.py` resolves them
  package-relatively via `importlib.resources` (works from a source checkout AND an
  installed wheel), `MANIFEST.in` includes them and `pyproject.toml`
  `[tool.setuptools.package-data]` ships them.
- The root `pyproject.toml` declares `evoxc` + `numpy` as runtime dependencies and sets
  `include-package-data = true` with `[tool.setuptools.packages.find] namespaces =
  true`, so the PEP 420 `evox`/`evox_ext` namespaces (and their data) build
  into the wheel.
- `evox.__version__` comes from that metadata, so it tracks the `pyproject.toml`
  version with no second source of truth.
Nothing under `src/evox/**` imports torch, and the sibling suite at
`../unit_test/evox/` is `evoxc`-only.

## Not ported from the 1.x line
Deliberately NOT ported from the 1.x implementation (removed in 2.0), with the concrete
reason (parallels live in the child CONTEXT.md files / `problems/hpo_wrapper.py`, not
here):
- torch `triton_kernels` — torch-dispatcher registration + hand-written Triton
  CUDA kernels have no `evoxc` analogue; the virtual-population noise is instead
  regenerated in-graph by `algorithms/so/es_variants/virtual_noise.py`
  (deterministic splitmix64 + Box-Muller).
- `problems/neuroevolution/{brax,mujoco_playground,supervised_learning,utils}.py`
  — bound to JAX/Brax/MuJoCo environments and to `torch.nn` + `DataLoader`; only
  the virtual Gaussian-noise problem is ported (see
  `problems/neuroevolution/CONTEXT.md`).
- `StdWorkflow` distributed multi-rank eval (`enable_distributed` / `group`) — no
  `evoxc` analogue; the `evoxc` workflow is single-device.
- torch core `ModuleBase`/`Parameter`/`Mutable`/`compile`/`vmap`/`use_state` —
  superseded by the functional design (config dataclasses + tensor state + plain
  functions; DESIGN.md §4).
- the HPO wrapper's in-graph nested-workflow path — EvoX Compiler has no eager mode and
  cannot nest `evoxc.run` inside a trace, so HPO is a host-side redesign in
  `problems/hpo_wrapper.py` where `HPOProblemWrapper` is NOT an `evox`
  `Problem` (see that module's docstring for the precise limitations).
