# evox — Functional EvoX on EvoX Compiler (Design Specification)

This document is the binding design spec for the evoxc-based functional rewrite of EvoX.
All implementers MUST follow it. Read it fully before writing code.

## 1. Goal

EvoX is a functional implementation of the EvoX evolutionary-computation framework
on top of the **EvoX Compiler tensor library** (the `evoxc` distribution, import
name `evoxc`), using plain functional code instead of the 1.x OOP design:

- **Config objects**: plain frozen dataclasses holding hyperparameters (Python scalars,
  strings, etc.). Never mutated, never traced.
- **State**: `namedtuple` / frozen `dataclass` **pytrees whose leaves are `evoxc` tensors
  only** (no Python scalars inside state).
- **Logic**: plain module-level functions (NOT `@evoxc.defn` — see §4.3) in the
  step protocol: `init`, `step`, `init_step`/`final_step`, `evaluate`, etc. No
  classes with `forward`, no `nn.Module`, no `Parameter`/`Mutable`.
- **Init is a separate function** (`init(config, key) -> state`) instead of a
  constructor that hides state creation. A step function owns the WHOLE
  generation: generate candidates → `fitness = evaluate(candidates)` → update
  the state (as in the 1.x `Algorithm.step`).

`evox` is the ONLY implementation in this repository: the 1.x torch tree was
removed in 2.0, so every rule below describes the shipped functional API and there is
no second implementation to compare against.

## 2. Hard facts about EvoX Compiler (design constraints)

These were verified empirically — do not re-investigate, do not fight them:

1. **EvoX Compiler has NO eager mode.** Every `evoxc.numpy`/`evoxc.ops`/`evoxc.random` call must happen
   inside an active trace (a function decorated `@evoxc.defn` called by `evoxc.trace`/
   `evoxc.build`/`evoxc.evaluate`/`evoxc.cond`/`evoxc.while_loop`/`evoxc.scan`). Calling an op
   outside a trace raises `evoxc.core.TraceError`.
   => `init` is ALSO traced, executed once through the pipeline
   (`evoxc.build` + `evoxc.run`).
2. **Graph inputs are tensors (and pytrees of tensors).** Python scalars CANNOT be
   graph inputs. Hyperparameters therefore live in the config dataclass and are
   passed as STATIC arguments of the traced functions (§4.3: `evoxc.build` treats
   every non-spec arg as a static pytree) — closures are fine during tracing.
   Scalars used in tensor expressions (e.g. `self.w * velocity`) become Python-float
   * tensor constant broadcasts — supported, but on compiler backends a mixed-dtype
   constant broadcast can fail; prefer explicit `evoxc.cast(constant, tensor.dtype)`
   when the op dtype is ambiguous.
3. **Pytrees**: `dict`, `list`, `tuple`, `namedtuple`, `dataclass` (+ defaultdict) flow
   as graph inputs/outputs and as loop carries. Use `evoxc.tree_map`/`tree_leaves`/
   `tree_flatten`/`tree_unflatten` (JAX-style names, in `evoxc` top level).
4. **RNG is stateless**: `evoxc.random.key(seed)`, `split(key)`, `split_n(key, n)`,
   `uniform(key, shape, low, high, dtype)`, `normal(key, shape, mean, std, dtype)`,
   `randint(key, shape, low, high, dtype)`, `permutation(key, n, dtype)`,
   `multinomial(key, input, num_samples)`. Keys never consumed; deterministic.
   Keys are **stored in the algorithm state** and advanced there (JAX-evoX style).
5. **Available ops** (in `evoxc` top level / `evoxc.numpy` as `enp`): elementwise math,
   `select/where`, `clamp/clip`, reductions (`sum/max/min/mean/prod/argmax/argmin`),
   `sort/argsort` (int64 indices), `topk`, `gather`, `scatter` (replacement only —
   no scatter-add), `cumsum/cumprod`, `dot/matmul`, `isnan`, `nan_to_num`, `median`,
   `nansum`, `var/std`, `reshape/transpose/broadcast/slice/concat/stack/tile/pad/flip/
   roll/diag/tril/triu/eye/linspace`, comparisons/logical, `cast`, `eigh/solve/norm`,
   `cond`, `while_loop`, `scan(f, init, xs, length=STATIC)`, `vmap` (axis 0 only).
   NO `einsum`. NO eager `jnp.ndarray.at[]` — use `scatter`.
   Naming gotchas: concatenation is `evoxc.concatenate` (NOT concat); no `squeeze` at
   top level; no `where`/`any`/`all`/`amax`/`amin`/`meshgrid`/`take`/`lexsort`/
   `repeat`/`moveaxis`/`atan2`; no scatter-add. See CONTEXT.md "EvoX Compiler issues found"
   for the full consolidated list of gotchas + workarounds.
6. **Compiler backends** (used for speed/GPU):
   - `"numpy"` (default interpreter — always works, CPU reference),
   - `"iree"` (llvm-cpu and cuda; cuda validated end-to-end for PSO/DE/OpenES/Sphere/
     DTLZ1-shaped graphs; use DEFAULT compile options: `while_init_rewrite` ON +
     `sort_emission` auto→count; cuda device ids are 1-based internally but the adapter
     maps `Device("cuda", N)` → N+1; NEVER use opt_level O2/O3 on cuda for eigh or
     NSGA2-selection graphs — default opt level is fine),
   - `"xla"` (PJRT plugin; CUDA via `jax_plugins/xla_cuda12/xla_cuda_plugin.so` in the
     shared venv — the xla adapter fixes are merged into `evoxc` master, see §9),
   - `"tvm"` (llvm cpu; no control flow — unsuitable for NSGA-family loops).
   Capabilities differ per backend; anything unsupported raises a loud
   `evoxc.core.BackendError`/`TransformError` (never silent fallback).
7. **vmap rules exist only for a subset of ops** (no vmap over sort/argsort/cumprod/
   nan_to_num...). Do NOT use vmap in algorithm/operator code — write batched tensor
   ops directly (populations are tensors with leading batch dim; that IS the batching).
8. **The run-loop pattern for speed**: build the step graph ONCE
   (`evoxc.build(step_fn, *specs, backend=..., ...)`, `evoxc.load(..., device=...)`), then
   run it repeatedly feeding outputs back as inputs (same-device loop). Per-step
   `evoxc.evaluate` (retrace+recompile) is the anti-pattern. On cuda, inputs must be
   explicitly placed via `t.to(evoxc.core.Device("cuda", N))` — no implicit transfers.

## 3. Package layout

```
src/evox/
├── __init__.py            # exports: core protocols, StdWorkflow, EvalMonitor,
│                          #   random re-export (as evoxc.random), key helpers
├── core/
│   ├── algorithm.py       # Algorithm protocol + type aliases (docstrings; no base class required)
│   ├── problem.py         # Problem protocol
│   ├── workflow.py        # StdWorkflow: composes init/step/evaluate, compiles per step variant
│   ├── monitor.py         # Monitor protocol
│   └── state.py           # State helpers: e.g. replace(), get/set nested (pytree utils)
├── algorithms/
│   ├── so/de_variants/  es_variants/  pso_variants/      # one module per algorithm
│   │   └── es_variants/virtual_{noise,es,lora_es}.py +   # deterministic splitmix64
│   │       _virtual_common.py                            #   noise + virtual-pop ES
│   └── mo/                                              # nsga2, nsga3, moead, rvea, rveaa, hype
├── operators/             # pure functions, one module per operator family
│   ├── sampling/    selection/    crossover/    mutation/
├── problems/numerical/    # basic.py, dtlz.py, cec2022.py
├── problems/neuroevolution/  # virtual_problem.py (VirtualProblem/VirtualLoRAProblem)
├── problems/hpo_wrapper.py   # host-side HPO (HPOProblemWrapper + random_search)
├── metrics/               # gd.py, hv.py, igd.py
├── workflows/             # std_workflow.py, eval_monitor.py
└── utils/                 # tree helpers, min_by, dominate_relation, pairwise_*_dist,
                           #   parse_opt_direction, rank, cal_max
```

## 4. Core API contract (binding)

### 4.1 Config dataclasses

Every algorithm/problem/monitor config is a **dumb frozen dataclass** (use
`@dataclasses.dataclass(frozen=True)`) whose field names and defaults ARE the public
API (`make_*` constructors keyword-map onto them), with these adjustments:
- There is no `device` field: the device is chosen per workflow (`StdWorkflow(device=...)`).
- Configs are dumb: no `__post_init__`, no normalization, no validation logic
  (the numerical-problem carve-out below is the only exception). They store ONLY
  plain static leaves — Python scalars (floats, ints, strings, `None`) and **flat
  `float` tuples**. Tuple storage is the package convention for array-like values:
  frozen `__eq__`/`__hash__` work on tuples (ndarray fields are unhashable), and
  `evoxc.run` performs strict by-value static revalidation on plain-leaf pytrees, so
  a drifted config value raises a TraceError instead of silently corrupting the
  run.
- `lb`/`ub` (boundary tensors in 1.x) are therefore **flat `float` tuples**
  of length `dim` in the config. Inside functions, bake them ONCE as graph
  constants:
  `lb = evoxc.ops.constant(evoxc.core.tensor(np.asarray(config.lb, dtype=np.float32)))`
  (closure-captured concrete tensors fail at trace time; constants are fine).
  `dim = len(config.lb)` stays a Python int (static).
- Optional operator fields (`selection_op: Optional[Callable]`, `mutation_op`,
  `crossover_op`) become **plain function references** (first-class functions —
  the operators are already pure functions, see §4.3); `None` means "algorithm
  default".

**Construction policy — `make_<algorithm>` constructors (binding):** for every
algorithm config, the array-accepting entry point is a module-level functional
constructor `make_<algorithm>` defined in the SAME module as its config dataclass —
the workflow resolves the algorithm module via
`importlib.import_module(type(config).__module__)` and must find the constructor
there, next to `init`/`step`. The constructor is where ALL normalization and
validation happens; the config dataclass itself stays dumb:
- ndarray arguments are converted to flat `float32` tuples (never stored as
  ndarrays);
- invalid input raises `ValueError` with a clear message (no bare asserts);
- defaults that must appear as fields are derived EAGERLY at construction — e.g.
  XNES/SeparableNES `pop_size` and learning rate from `dim`, ASEBO `subspace_dims`
  from `center_init` (`core/workflow.py` `_discover_pop_size` reads `cfg.pop_size`,
  so such fields must be filled in, not left `None`).
Direct dataclass construction with already-normalized plain statics remains
possible; array arguments go through `make_<algorithm>` only.

**ndarray statics and pytree registration:** ndarray values ARE legal static
leaves on the current `evoxc` master (commit b8062a9 "accept np.ndarray as static
trace values"), so the old rejection and its workarounds are obsolete — with ONE
exception: non-`None` callable fields are still rejected as static leaves anywhere
in the pytree. Zero-child `evoxc.register_pytree_node` registration is therefore
reserved for configs carrying callable op fields — `mo/nsga3`, `moead`, `rvea`
only. Every other config is a plain pytree and gets `evoxc`'s by-value static
revalidation at `evoxc.run` (an opaque registered node is NOT revalidated).

**Carve-out — numerical problems (`problems/numerical/basic.py`,
`cec2022.py`):** these KEEP their validation-only `__post_init__` methods as the
sanctioned direct-construction public API — no ndarray→tuple normalization there
(`shift`/`affine` are stored as passed, validated only), and their clear
`AssertionError` messages (bare asserts with message strings) are asserted by
the unit tests — the messages are frozen, so do NOT convert them to
`ValueError`. The `make_*` pattern does not apply to them.

### 4.2 State

- Each algorithm module defines a frozen dataclass (or namedtuple) `<Name>State`
  with a leaf per mutable quantity, e.g. `PSOState(population, velocity,
  local_best_location, local_best_fitness, global_best_location, global_best_fitness,
  key)`. All leaves are `evoxc` tensors (float32 preferred for fitness/population;
  int32 for indices; keys are whatever `random.split` returns).
- `WorkflowState(algorithm_state, problem_state, monitor_state, generation, key)` —
  a dataclass in `core/workflow.py` (monitor_state may be `None` → use a constant
  dummy tensor leaf).
- Problem state: numerical problems are stateless → `ProblemState()` empty frozen
  dataclass (still a distinct type so the workflow can route it).
- Generation counter is an `int32` scalar tensor.
- Fitness dtype float32. Minimization semantics internally everywhere: the workflow
  applies `opt_direction` to convert maximization problems.

### 4.3 Functions (plain Python functions — NOT `@evoxc.defn`)

VERIFIED evoxc contract: `evoxc.trace`/`evoxc.build` accept plain callables; a `Defn`
object RAISES when called (even during another trace), so defn-wrapping would
break composition. Therefore **all algorithm/problem/monitor/operator functions
are PLAIN functions** (a bare call outside a trace raises `evoxc`'s "No active
trace" TraceError — that is fine and expected). The workflow traces them
explicitly.

```python
# algorithms/<...>/<name>.py  — module-level functions + config dataclass
init(config, key) -> State                          # draw initial state
step(config, state, evaluate) -> State              # REQUIRED: one full generation
# optional, when the first/last generation differs (NSGA-style full-population
# evaluation; the workflow falls back to `step` when absent):
init_step(config, state, evaluate) -> State
final_step(config, state, evaluate) -> State
#
# `step` owns the WHOLE generation: generate the candidate batch, obtain its
# fitness via `fitness = evaluate(candidates)` (may call it several times with
# different candidate sets), then update the state and return it. Candidates
# are (n, dim) float32; fitness is (n,) or (n, n_obj) float32 (min semantics).
# `evaluate` is a traced closure created by the workflow (§4.4); treat it as
# OPAQUE — do not store it in the state or re-thread it through returns; the
# workflow owns the problem/monitor state and the generation counter.

# problems/<...>.py
evaluate(config, problem_state, pop) -> (fitness, problem_state)   # pop: (n, dim)
# numerical problems are stateless but keep the signature for uniformity.

# operators/<...>.py — pure functions (called inside traces)
# RULE (binding): these names, argument names and order ARE the public API.
# - Functions that draw randomness gain `key` as the FIRST parameter.
# - There is no `device` parameter (evoxc has no device arg).
# - Everything operates on evoxc tensors (where->select, argsort->argsort,
#   gather->gather, clamp->clamp, etc.).
# Operator signatures:
#   simulated_binary(x, pro_c, dis_c)                -> simulated_binary(key, x, pro_c, dis_c)
#   simulated_binary_half(x, pro_c, dis_c)           -> + key first
#   DE_differential_sum(diff_padding_num, num_diff_vects, index, population, F=None,
#                       replace=False)               -> + key first (F=None selects the default)
#   DE_binary_crossover(mutation_vector, current_vector, CR)         -> + key
#   DE_exponential_crossover(mutation_vector, current_vector, CR)    -> + key
#   DE_arithmetic_recombination(mutation_vector, current_vector, K)  -> unchanged (no rng)
#   polynomial_mutation(x, lb, ub, pro_m, dis_m)     -> polynomial_mutation(key, x, lb, ub, pro_m, dis_m)
#   latin_hypercube_sampling_standard(n, d, smooth)  -> + key, NO device arg
#   latin_hypercube_sampling(n, lb, ub, smooth)      -> + key
#   uniform_sampling(n, m)                           -> unchanged (Das-Dennis, deterministic)
#   grid_sampling(n, m)                              -> unchanged
#   tournament_selection(n_round, fitness, tournament_size=2)        -> + key first
#   tournament_selection_multifit(n_round, fitnesses, tournament_size=2) -> + key first
#   select_rand_pbest(percent, population, fitness)  -> + key first
#   dominate_relation(x, y) / non_dominate_rank(x) / crowding_distance(costs, mask) /
#   nd_environmental_selection(x, f, topk) / apd_fn(...) / ref_vec_guided(x, f, v, theta)
#                                                   -> unchanged (no rng)

# metrics/<...>.py
igd(objs, pf, p=1.0) -> scalar tensor            # etc. — plain functions
```

**Static config args — no partialization needed (VERIFIED):** `evoxc.trace`/
`evoxc.build` treat each positional arg as one pytree: `TensorSpec` leaves become
tensor inputs, everything else (config dataclasses, ints, floats, strings)
becomes a STATIC value passed as-is into the function (static Python control
flow over them specializes the graph). So:

```python
exe = evoxc.build(module.step, algo_config, state_spec_tree, backend=..., device=...)
state = evoxc.run(exe, algo_config, state)   # run() accepts/returns full pytrees
```

**Module convention (how the workflow finds the functions):** the `init`/
`step` (and optional `init_step`/`final_step`) functions live in the SAME
module as their config dataclass. The workflow resolves them via
`importlib.import_module(type(config).__module__)` (a module defining
`step` but no `init_step`/`final_step` falls back to `step` for the
first/last generation). Same for problems
(`evaluate`) and monitors (`monitor_update`).

**Constants:** bake constant tensors inside functions with `evoxc.ops.constant`
(numpy array arg) — closure-captured CONCRETE tensors fail at trace time.
Python scalars in expressions are fine (constant broadcast).

**Key management convention** (JAX-evoX style): every `init`/`step` (and
variant) that needs randomness splits the key FROM THE STATE:
`key, subkey = random.split(state.key)`,
uses subkeys, and stores `key` back. Step functions must be deterministic
given the state (draw all randomness before the `evaluate` call).

### 4.4 StdWorkflow (`core/workflow.py`, `workflows/std_workflow.py`)

Plain (non-frozen) class holding: algorithm config, problem config, monitor config
(or None), `opt_direction` ("min"/"max"/list), `num_generations`, backend name,
device, compile options. Public methods:

```python
wf = StdWorkflow(algorithm=..., problem=..., monitor=..., opt_direction="min", ...)
wf.init(seed=42)                      # builds init graph, runs once -> initial state
wf.init_step(state)                   # first generation (falls back to step)
wf.step(state)                        # one generation (compiled executable run)
wf.final_step(state)                  # last generation (falls back to step)
wf.run(generations, seed=42) -> state # init + loop; collects monitor history
wf.fit(fitness=..., generations=...)  # in-place fit helper
```

Implementation sketch (binding):
```python
step_fn = _make_step_fn(variant)  # composes fn(algo_cfg, alg_state, evaluate);
                                  # builds the opaque evaluate closure (below)
init_exe = evoxc.build(init_body, TensorSpec((), np.int64))       # cpu, numpy backend
state = evoxc.run(init_exe, key)                                   # once
state = tree_map(lambda t: t.to(device), state)                  # if device is cuda
specs = evoxc.tree_map(lambda t: TensorSpec(t.shape, t.dtype), state)
exe = evoxc.build(step_fn, specs, backend=backend, device=device, **opts)
                                  # per (variant, state signature), lazily;
                                  # re-traced when tensor-leaf shapes/dtypes drift
# per generation (same-device loop, pytrees in/out):
state = evoxc.run(exe, state)
# host-side monitor history: after each step, copy out monitor state leaves
#    via leaf.to(evoxc.core.Device("cpu")).numpy() and append to Python lists.
```

The composed step (inside one graph): the algorithm's step-family function runs
the whole generation, calling the workflow-owned opaque closure
```
evaluate(candidates) -> fitness
```
which applies, in order: solution_transform → `problem.evaluate` (threading the
problem state) → opt-direction scaling (min semantics) → fitness_transform →
monitor update (threading the monitor state; the monitor sees the RAW candidates
and the TRANSFORMED fitness), then the workflow bumps the generation counter.
The FIRST/LAST-generation dispatch (`init_step`/`final_step` vs `step`) is a
HOST-SIDE decision — the workflow builds one executable per
(variant, state-signature) lazily (re-tracing when tensor-leaf shapes/dtypes
drift, e.g. CoDE's `(3n, dim)` or CSO's `(n/2, dim)` monitor batches) and
picks the right one per generation; there is no in-graph
`evoxc.cond(generation == 0)` branch. Every quantity stays in-graph;
Python-level history recording happens in the workflow loop AFTER each run.

**Non-tensor candidate payloads (virtual ES family).** The opaque `evaluate`
closure accepts a NON-tensor candidate PAYLOAD in addition to the `(n, dim)`
tensor path: the virtual-population ES family
(`algorithms/so/es_variants/virtual_es.py`, `virtual_lora_es.py`) calls
`evaluate((center, seeds, sigma))` — `center` `(dim,)` f32, `seeds` `(pop,)`
int64, `sigma` a static Python float — and the problem
(`problems/neuroevolution/virtual_problem.py`) unpacks that payload to
regenerate each individual's noise on demand. The 2-D `(n, dim)` tensor path is
UNCHANGED; the payload flows through the same pipeline (the static `sigma` leaf
is specialized by value like any other static, §4.3) and the problem returns
`(pop,)` fitness with minimization semantics. `solution_transform`/
`fitness_transform` are applied by the closure exactly as for the tensor path.

### 4.5 Monitor (`core/monitor.py`, `workflows/eval_monitor.py`)

`MonitorState` dataclass holds tensor leaves (e.g. latest solution/fitness, elite
buffers of fixed size). `monitor_update(config, state, candidate, fitness) -> state`
is a `@evoxc.defn`. Host-side history lists (solution_history, fitness_history) live on
the monitor CONFIG object (plain Python lists) and are appended by the workflow
(post-step host copy). `EvalMonitor` must support both single- and multi-objective
(use `topk`/argsort for SO elite; non-dominated rank for MO Pareto front) and expose
`get_best_solution`/`get_best_fitness`/`plot`.
`plot(problem_pf=None, source="eval", **kwargs)` is implemented via the ported host-side
`evox.vis_tools` (returns a plotly `Figure`; warns + returns `None` when plotly is
unavailable or no history was recorded).
Algorithms MAY expose an optional host-side `record_step(config, state, candidate,
fitness) -> dict` hook; the workflow appends its dict to the monitor's `aux_history`
(gated on `full_pop_history`), and `plot(source="pop")` reads `aux_history["fit"]`.

## 5. Rules for algorithm implementations

1. The math is fixed by the algorithm's published definition; `bound_method` handling,
   batch processing, mirrored sampling etc. are part of the semantics and must be
   preserved.
2. Op mapping used throughout the package: `where` → `evoxc.select` (or `enp.where`),
   `clamp` → `evoxc.clamp`, `argsort` → `evoxc.argsort`, `gather` → `evoxc.gather`,
   index writes → `evoxc.scatter`, `cumsum` → `evoxc.cumsum`, matmul → `evoxc.dot`.
   Index writes go through `evoxc.scatter` (replacement only — never accumulate; an
   accumulating update must be restructured to compute sums first, then scatter).
3. Eigen-decomposition → `evoxc.eigh` (CMA-ES).
4. There is no per-function `compile`/`vmap` annotation — the whole step is one
   compiled graph.
5. RNG: `evoxc.random.*` with keys split from `state.key`. Determinism: same seed →
   same trajectory (within fp32 tolerance) across backends.
6. State dataclass field naming follows the 1.x `State` attribute names, which are the
   public state vocabulary.
7. Configs are frozen dataclasses; if an algorithm truly needs mutable aux data
   (e.g. monitor history), it lives on the workflow/monitor config (host-side).
8. No Python control flow depending on tensor VALUES inside a trace (use `evoxc.cond`/
   `select`). Static Python control flow on shapes/config is fine.
9. Numerical problems: `evaluate` = vectorized math over the (n, dim) tensor (never a
   Python loop over n). CEC2022: the formulas AND the input-data loading
   (cec2022_input_data as numpy arrays loaded via importlib.resources; input data is
   CONSTANT tensors baked via closure, not graph inputs).
10. If an algorithm genuinely cannot be expressed with the ops `evoxc` provides and no
    composition exists, DO NOT fake it: leave it out and report it. (All 34 algorithms
    are implemented today.)

## 6. Testing strategy

`unit_test/evox/` mirrors the package and is the only test suite (numpy backend):
- **Correctness**: every algorithm runs a real end-to-end optimization and asserts the
  final best fitness reaches the expected basin (tolerance 1e-3 relative for
  convergence-sensitive tests; exact-determinism tests compare against `evoxc` numpy
  backend golden values).
- **Reproducibility**: the same integer seed reproduces the same trajectory, and
  changing the seed changes it.
- Run with `PYTHONPATH=src <venv>/bin/python -m pytest -q` from the repo root (venv
  recipe in `src/evox/CONTEXT.md` "Environment").
- IMPORTANT: no test file imports torch — the suite is `evoxc`-only.

## 7. Benchmarks

`benchmarks/evox_vs_torch/` — the backend comparison harness (see the CONTEXT.md there;
the `evox_vs_torch` directory name and the `evox-*` backend ids are frozen keys into the
committed result JSONs):
- SO: PSO, DE, CMA-ES, OpenES on Sphere/Rastrigin/Ackley; MO: NSGA2 (+ NSGA3, MOEAD)
  on DTLZ1/2.
- Scales: small (pop=100, dim=10), medium (pop=1000, dim=50), large (pop=10000,
  dim=100) — SO; (pop=100/1000, 3 obj, dim=10/30) — MO. 100 generations.
- Runners: `evox-numpy`, `evox-iree-llvm-cpu`, `evox-iree-cuda`, `evox-iree-rocm`,
  `evox-xla-cuda`, `evox-xla-rocm` — same seeds/algorithms. Output: table of wall-time
  per step + total, fitness convergence, and per-case HV/IGD with a comparison
  verdict against the frozen `results/{suite}_torch-cpu.json` baseline (1.x
  measurements, read as JSON only).

## 8. Environment

Build a venv with `uv venv` + `uv pip install -e . pytest plotly pandas` (`evoxc` and
`numpy` come from `pyproject.toml`; the compiled backends need the matching extras
`evoxc[iree]`/`evoxc[tvm]`, and XLA needs a PJRT plugin found through
`EVOXC_PJRT_PLUGIN`), then run tests from it with `PYTHONPATH=src`.
`evox` and `evox_ext` are PEP 420 namespace packages, so `PYTHONPATH=src` is
required in a source checkout. Lint with `ruff check src/evox src/evox_ext`.
GPUs: on a GPU host, scan `nvidia-smi` for the most-free GPU before GPU runs.

## 9. EvoX Compiler repo fixes (merged into `evoxc` master)

The xla-related `evoxc` fixes below are ALL merged into `evoxc` master (@f2f50a7) — the
test venv's `evoxc` install needs no task branch:
- xla adapter GPU client-exhaustion fix (fresh PJRT client per compile/load →
  process SIGABRT; now a shared refcounted client — `evoxc` commit 838739c, formerly
  only on task branch `evogit-agent-T1-A19`);
- `GetPjRtApi` symbol-casing fallback + serialized `CompileOptionsProto` handling
  (`evoxc` commit 71d721e);
- the xla rank-0 staging fixes.
evox code must NOT depend on any of these fixes — it only needs the `evoxc` API
already on master. If you hit an `evoxc` bug/missing feature, record it in
`src/evox/CONTEXT.md` under "EvoX Compiler issues found" (don't try to fix
`evoxc` yourself — escalate to the root agent).

## 10. Style rules

- Type hints on all public functions. Docstrings 1-3 lines.
- No classes with methods that mutate tensors. Functions only.
- Keep files < ~500 lines; keep the 1.x file names so the layout stays familiar.
- No torch imports anywhere in `src/evox/**`; numpy is allowed only host-side
  (`problems/numerical/cec2022.py` input-data loading, `make_*` preprocessing, metrics,
  HPO, visualization) and never in graph code.
- `from evoxc import tree_map, ...` and `import evoxc.numpy as enp`, `import evoxc.random as
  random` are the sanctioned imports. `evoxc` top level re-exports most ops.
