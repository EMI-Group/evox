# evox_etl — Functional EvoX on EvoX Compiler

## Intent
A functional rewrite of the EvoX framework on the EvoX Compiler tensor library
(`evoxc`, installed from its source checkout into a locally built test venv —
see "Environment" below). Replaces the
torch OOP design (`src/evox/`, kept untouched as reference) with plain functional code:
frozen config dataclasses, namedtuple/dataclass tensor states, plain module-level
functions (no `@evoxc.defn` — see DESIGN.md §4.3) in the step protocol
(`init(config, key) -> state`, `step(config, state, evaluate) -> state`), and a
compile-once StdWorkflow loop.
**Binding spec: `DESIGN.md` in this directory — read it fully before writing any code.**

## API Surface
- `evox_etl.core`: `Algorithm`/`Problem`/`Monitor` protocols (duck-typed), `StdWorkflow`,
  `WorkflowState`, state helpers.
- `evox_etl.algorithms`: SO (de/es/pso variants, incl. the training-free `virtual_es`/
  `virtual_lora_es` family) + MO (nsga2, nsga3, moead, rvea,
  rveaa, hype) — each module = config dataclass + `make_*` constructor +
  `init`/`init_step`/`step` plain functions (step protocol, see
  `core/algorithm.py`). Public surface mirrors torch: `VirtualES`/`VirtualLoRAES`
  are the bare torch names (both alias the VirtualES config), the distinct low-rank
  config is `VirtualLoRAESConfig`. The virtual-population ES family adds
  `virtual_es.py` (`VirtualES`/`VirtualLoRAES` + `make_virtual_es`),
  `virtual_lora_es.py` (`VirtualLoRAESConfig` + `make_virtual_lora_es`) and the
  shared deterministic-noise module `virtual_noise.py`.
- `evox_etl.operators`: pure functions (sampling, selection, crossover, mutation).
- `evox_etl.problems.numerical`: basic, dtlz, cec2022.
- `evox_etl.problems.neuroevolution`: `VirtualProblem`/`VirtualLoRAProblem`
  (unified behind `lora_rank`) + `make_virtual_problem` + `evaluate`
  (payload protocol `(center_flat, seeds, sigma)`).
- `evox_etl.problems.hpo_wrapper`: HOST-side HPO (`HPOProblemWrapper`, `HPSlot`,
  HPO monitor configs, `random_search`) — re-exported from `evox_etl.problems`
  (NOT an `evox_etl` `Problem`).
- `evox_etl.metrics`: igd, gd, hv. `evox_etl.workflows`: std_workflow, eval_monitor.
- `evox_etl.utils`: tree helpers, min_by, dominate_relation, pairwise distances,
  parse_opt_direction, rank.

## Constraints
- NO eager tensor ops — everything inside `@evoxc.defn`/traces (EvoX Compiler has no eager mode).
- Config fields are plain static leaves (Python scalars + flat float tuples)
  partialized away at compile time; state leaves are tensors ONLY. Minimization
  semantics internally (workflow applies opt_direction).
- No torch/numpy in framework code (numpy allowed only to load cec2022 input data and
  for host-side make_* preprocessing).
- EvoX Compiler issues found while implementing: record under "EvoX Compiler issues
  found" below and escalate to the root agent (do NOT edit the `evoxc` source checkout
  at `<path-to-evoxc-checkout>` from here).

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
| Metrics | `metrics/` | gd/gd_plus, igd/igd_plus, hv + MC variants; tests at `../unit_test/etl/metrics/` |
| Workflow + EvalMonitor | `workflows/` | std_workflow re-export, eval_monitor; tests at `../unit_test/etl/workflows/` |
| Visualization | `vis_tools/` | Plotly figure builders + EvoXVision `.exv` serialization; tests at `../unit_test/etl/vis_tools/` |
| Utilities | `utils/` | functional helpers |
| End-to-end tests (e2e + autoload) | `../unit_test/etl/` | real-algorithm convergence + autoload-guard tests (mirror) |
| Tests | `../unit_test/etl/` | sibling — mirrors this package |
| Benchmarks (torch vs evoxc) | `../benchmarks/etl_vs_torch/` | sibling — comparison harness |
| Reference (torch) impl | `../evox/` | sibling — READ-ONLY, never modify |
| EvoX Compiler library | `<path-to-evoxc-checkout>` | read via the installed `evoxc` package; fixes only by root agent |

## Environment
No pre-provisioned venv — build one locally with `uv venv` +
`uv pip install <path-to-evoxc-checkout> numpy pytest "torch>=2.6.0"`
(`<path-to-evoxc-checkout>` is the EvoX Compiler source checkout whose
`pyproject.toml` has `name = "evoxc"`; a
published `evoxc` distribution can be installed by bare name instead; torch-cpu wheel
index suffices — no GPU here). The primary
`/home/bill/Source/evox/.venv` is read-only and lacks evoxc/pytest, so run the
gate from the locally built test venv with
`PYTHONPATH=src:/home/bill/Source/evox/.venv/lib/python3.13/site-packages`
(that primary venv supplies plotly/torch). Canonical ETL suite: the sibling
`../unit_test/etl/` (mirrors this package) —
`<venv>/bin/python -m pytest unit_test/etl -q` (numpy backend; ~6 min,
618 tests; per-suite counts in `../unit_test/etl/CONTEXT.md`).
`evox_etl` and `evox_etl_ext` are PEP 420 namespace packages — drivers need
`PYTHONPATH=src`.

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
   `__post_init__`s (AssertionError with message, frozen because `../unit_test/etl`
   asserts them — do NOT convert to ValueError).
2. **`evoxc.select` does not numpy-broadcast a `(n,)` condition against `(n, m)`
   branches** (`cannot broadcast incompatible dims n and m`). numpy/torch broadcast
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
   align over k (use expand_dims); `evoxc.svd` returns reduced (U,S,Vh) matching
   torch svd(some=True).
7. **No scatter-add** (`evoxc` `scatter` is replacement-only put_along_axis) —
   worked around with one-hot segment sums (mo/nsga3); a composition note, not a bug.
8. Cosmetic: the `evoxc` numpy backend leaks a `RuntimeWarning: invalid value
   encountered in divide` for the intentional inf-beta draws (SBX/SHADE/SaDE NaN
   paths) — torch-identical semantics, no change made.
### Additional findings (core/operators/problems teams)
9. **Inconsistent axis arg naming**: reductions take `axes=` but topk/argmin/gather/
   `enp.sum/min` take `axis=`. Check the op signature before using.
10. `evoxc.tree_unflatten(leaves, treespec)` argument order is REVERSED vs JAX.
11. Top-level `evoxc.zeros/ones/full/empty` are EAGER concrete creators (raise inside
    traces); in-graph constants: `evoxc.ops.constant(evoxc.core.tensor(np.asarray(...)))`
    — `constant()` rejects raw ndarray; `x * 0.0` for zero-init; no `*_like` ops;
    `enp.floor` missing.
12. Dtype promotion differs from torch: int32+int→int64, int32*float→float64,
    float32**int64→float64 — explicit `evoxc.cast` needed.
13. `TensorSpec.from_tensor` does NOT exist (flat `(shape, dtype)` only);
    `Executable` has no `.run` (use `evoxc.run(exe, *args)`); `evoxc.evaluate` rejects
    static args (use `evoxc.build`); scalar tensor inputs must be 0-d ndarrays, not
    numpy scalars. Statics are path-specialized per distinct value — a different
    static value at run raises TraceError (build a new exe per value).
14. `evoxc.gather` is numpy-take semantics (not torch gather); take_along_axis needs a
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
`algorithms/__init__.py` (torch-style bare-name aliases unchanged).
Carve-out: numerical problems (basic.py, cec2022.py) keep their validation-only
`__post_init__`s — direct dataclass construction is their sanctioned public API.
Unit-test construction sites at `../unit_test/etl` migrate to `make_*` in a
parallel wave (direct construction still works for pre-normalized values but
bypasses make_* validation).

## Public-surface deltas vs torch `evox` (current state, matters for migration)
- Root `evox_etl` re-exports core protocols, `StdWorkflow`, `EvalMonitor(Config)`,
  the 5 metric functions and `random`/`key`; it does NOT explicitly import
  `algorithms`/`problems`/`operators` — those become package attributes only because
  `auto_load_extensions()` imports the 5 domains as a side effect.
- `evox_etl.utils` is NOT the torch `evox.utils` surface: torch's
  `switch/clamp*/clip/maximum*/minimum*/lexsort/nanmin/nanmax/randint` live in
  `operators/jit_fix_operator.py` and are re-exported from neither `evox_etl.utils`
  nor `evox_etl.operators`; `ParamsAndVector`, `VmapInfo`/`register_vmap_op` have no
  counterpart (virtual problems take pre-flattened `param_shapes`/`layer_specs`).
- Bare torch algorithm names resolve at `evox_etl.algorithms` and
  `.algorithms.so`; `.algorithms.mo` and `.so.es_variants` expose only
  `*Config` + `make_*`.
- `metrics.hv` takes a leading RNG key (`hv(key, objs, ref, num_sample)`); the port
  adds `gd_plus`/`igd_plus`/`bounding_cube_monte_carlo_hv`/`each_cube_monte_carlo_hv`.
- DTLZ exposes the Pareto front as module-level `pf(config)`/`pf_dtlzN`, not a
  `DTLZ.pf()` method; there is no `DTLZ` base class.
- `vis_tools` exports all `plot_*` + `EvoXVisionAdapter`/`new_exv_metadata` at package
  level (torch `evox.vis_tools.__init__` is empty) and imports Plotly lazily.

## Known release blockers (torch deletion)
- `problems/numerical/cec2022.py:27` resolves `DATA_DIR` to
  `.../src/evox/problems/numerical/cec2022_input_data` — the tree being deleted (and
  an invalid path from an installed wheel); no data or `package-data` exists under
  `evox_etl`, `MANIFEST.in` lists only `src/evox/...`, and
  `../unit_test/etl/problems/test_cec2022.py` has no skip.
- `pyproject.toml` is still the 1.x manifest: `dependencies = ["torch>=2.6.0", ...]`
  with no `evoxc`, MANIFEST only `src/evox/...`; the `evox_ext`/`evox_etl_ext` PEP 420
  namespaces have no `__init__.py`, so verify they are included by the wheel build.
- 15 `unit_test/etl/**/parity/**` modules import torch and the torch `evox` package
  (no pytest marker/deselect config); `examples/hpo/*.py`, 6 `benchmarks/*` scripts
  and the whole `docs/` tree (0 `evox_etl` references) still target the torch API.

## Not portable from torch evox
Deliberately NOT ported, with the concrete reason (parallels live in the child
CONTEXT.md files / `problems/hpo_wrapper.py`, not here):
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
  `problems/hpo_wrapper.py` where `HPOProblemWrapper` is NOT an `evox_etl`
  `Problem` (see that module's docstring for the precise limitations).
