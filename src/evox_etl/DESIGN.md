# evox_etl — Functional EvoX on ETL (Design Specification)

This document is the binding design spec for the ETL-based functional rewrite of EvoX.
All implementers MUST follow it. Read it fully before writing code.

## 1. Goal

Build a new, functional implementation of the EvoX evolutionary-computation framework
on top of the **ETL tensor library** (foreign repo `/mnt/local-ssd/bchuang/etl`, import
name `etl`), replacing the torch OOP design with plain functional code:

- **Config objects**: plain frozen dataclasses holding hyperparameters (Python scalars,
  strings, etc.). Never mutated, never traced.
- **State**: `namedtuple` / frozen `dataclass` **pytrees whose leaves are ETL tensors
  only** (no Python scalars inside state).
- **Logic**: pure functions decorated with `@etl.defn` — `init`, `ask`, `tell`,
  `evaluate`, etc. No classes with `forward`, no `nn.Module`, no `Parameter`/`Mutable`.
- **Init is a separate function** (`init(config, key) -> state`) instead of a
  constructor that hides state creation.

The existing torch-based `evox` package (v1.3.0, in `src/evox/`) stays untouched and
serves as the reference implementation for parity tests and benchmarks.

## 2. Hard facts about ETL (design constraints)

These were verified empirically — do not re-investigate, do not fight them:

1. **ETL has NO eager mode.** Every `etl.numpy`/`etl.ops`/`etl.random` call must happen
   inside an active trace (a function decorated `@etl.defn` called by `etl.trace`/
   `etl.build`/`etl.evaluate`/`etl.cond`/`etl.while_loop`/`etl.scan`). Calling an op
   outside a trace raises `etl.core.TraceError`.
   => `init` is ALSO a `@etl.defn` function, executed once through the pipeline
   (`etl.evaluate` or an executable run).
2. **Graph inputs are tensors (and pytrees of tensors).** Python scalars CANNOT be
   graph inputs. Hyperparameters therefore live in the config dataclass and are
   **partialized away** at compile time:
   `functools.partial(module.ask, config)` — closures are fine during tracing.
   Scalars used in tensor expressions (e.g. `self.w * velocity`) become Python-float
   * tensor constant broadcasts — supported, but on compiler backends a mixed-dtype
   constant broadcast can fail; prefer explicit `etl.cast(constant, tensor.dtype)`
   when the op dtype is ambiguous.
3. **Pytrees**: `dict`, `list`, `tuple`, `namedtuple`, `dataclass` (+ defaultdict) flow
   as graph inputs/outputs and as loop carries. Use `etl.tree_map`/`tree_leaves`/
   `tree_flatten`/`tree_unflatten` (JAX-style names, in `etl` top level).
4. **RNG is stateless**: `etl.random.key(seed)`, `split(key)`, `split_n(key, n)`,
   `uniform(key, shape, low, high, dtype)`, `normal(key, shape, mean, std, dtype)`,
   `randint(key, shape, low, high, dtype)`, `permutation(key, n, dtype)`,
   `multinomial(key, input, num_samples)`. Keys never consumed; deterministic.
   Keys are **stored in the algorithm state** and advanced there (JAX-evoX style).
5. **Available ops** (in `etl` top level / `etl.numpy` as `enp`): elementwise math,
   `select/where`, `clamp/clip`, reductions (`sum/max/min/mean/prod/argmax/argmin`),
   `sort/argsort` (int64 indices), `topk`, `gather`, `scatter` (replacement only —
   no scatter-add), `cumsum/cumprod`, `dot/matmul`, `isnan`, `nan_to_num`, `median`,
   `nansum`, `var/std`, `reshape/transpose/broadcast/slice/concat/stack/tile/pad/flip/
   roll/diag/tril/triu/eye/linspace`, comparisons/logical, `cast`, `eigh/solve/norm`,
   `cond`, `while_loop`, `scan(f, init, xs, length=STATIC)`, `vmap` (axis 0 only).
   NO `einsum`. NO eager `jnp.ndarray.at[]` — use `scatter`.
   Naming gotchas: concatenation is `etl.concatenate` (NOT concat); no `squeeze` at
   top level; no `where`/`any`/`all`/`amax`/`amin`/`meshgrid`/`take`/`lexsort`/
   `repeat`/`moveaxis`/`atan2`; no scatter-add. See CONTEXT.md "ETL issues found"
   for the full consolidated list of gotchas + workarounds.
6. **Compiler backends** (used for speed/GPU):
   - `"numpy"` (default interpreter — always works, CPU reference),
   - `"iree"` (llvm-cpu and cuda; cuda validated end-to-end for PSO/DE/OpenES/Sphere/
     DTLZ1-shaped graphs; use DEFAULT compile options: `while_init_rewrite` ON +
     `sort_emission` auto→count; cuda device ids are 1-based internally but the adapter
     maps `Device("cuda", N)` → N+1; NEVER use opt_level O2/O3 on cuda for eigh or
     NSGA2-tell graphs — default opt level is fine),
   - `"xla"` (PJRT plugin; CUDA via `jax_plugins/xla_cuda12/xla_cuda_plugin.so` in the
     shared venv — the xla adapter fixes are merged into etl master, see §9),
   - `"tvm"` (llvm cpu; no control flow — unsuitable for NSGA-family loops).
   Capabilities differ per backend; anything unsupported raises a loud
   `etl.core.BackendError`/`TransformError` (never silent fallback).
7. **vmap rules exist only for a subset of ops** (no vmap over sort/argsort/cumprod/
   nan_to_num...). Do NOT use vmap in algorithm/operator code — write batched tensor
   ops directly (populations are tensors with leading batch dim; that IS the batching).
8. **The run-loop pattern for speed**: build the step graph ONCE
   (`etl.build(step_fn, *specs, backend=..., ...)`, `etl.load(..., device=...)`), then
   run it repeatedly feeding outputs back as inputs (same-device loop). Per-step
   `etl.evaluate` (retrace+recompile) is the anti-pattern. On cuda, inputs must be
   explicitly placed via `t.to(etl.core.Device("cuda", N))` — no implicit transfers.

## 3. Package layout (mirrors torch evox arrangement)

```
src/evox_etl/
├── __init__.py            # exports: core protocols, StdWorkflow, EvalMonitor,
│                          #   random re-export (as etl.random), key helpers
├── core/
│   ├── algorithm.py       # Algorithm protocol + type aliases (docstrings; no base class required)
│   ├── problem.py         # Problem protocol
│   ├── workflow.py        # StdWorkflow: composes init/ask/tell/evaluate, compiles step
│   ├── monitor.py         # Monitor protocol
│   └── state.py           # State helpers: e.g. replace(), get/set nested (pytree utils)
├── algorithms/
│   ├── so/de_variants/  es_variants/  pso_variants/      # same files as torch evox
│   └── mo/                                              # nsga2, nsga3, moead, rvea, rveaa, hype
├── operators/             # pure functions, same file names as torch evox
│   ├── sampling/    selection/    crossover/    mutation/
├── problems/numerical/    # basic.py, dtlz.py, cec2022.py
├── metrics/               # gd.py, hv.py, igd.py
├── workflows/             # std_workflow.py, eval_monitor.py
└── utils/                 # tree helpers, min_by, dominate_relation, pairwise_*_dist,
                           #   parse_opt_direction, rank, cal_max
```

## 4. Core API contract (binding)

### 4.1 Config dataclasses

Every algorithm/problem/monitor config is a **dumb frozen dataclass** (use
`@dataclasses.dataclass(frozen=True)`) with the SAME field names and defaults as the
torch evox class `__init__` signature (read `src/evox/...` for the reference), with
these adjustments:
- Drop the `device: torch.device | None = None` parameter.
- Configs are dumb: no `__post_init__`, no normalization, no validation logic
  (the numerical-problem carve-out below is the only exception). They store ONLY
  plain static leaves — Python scalars (floats, ints, strings, `None`) and **flat
  `float` tuples**. Tuple storage is the package convention for array-like values:
  frozen `__eq__`/`__hash__` work on tuples (ndarray fields are unhashable), and
  `etl.run` performs strict by-value static revalidation on plain-leaf pytrees, so
  a drifted config value raises a TraceError instead of silently corrupting the
  run.
- `lb`/`ub` (boundary tensors in torch) therefore become **flat `float` tuples**
  of length `dim` in the config. Inside functions, bake them ONCE as graph
  constants:
  `lb = etl.ops.constant(etl.core.tensor(np.asarray(config.lb, dtype=np.float32)))`
  (closure-captured concrete tensors fail at trace time; constants are fine).
  `dim = len(config.lb)` stays a Python int (static).
- Optional operator fields (`selection_op: Optional[Callable]`, `mutation_op`,
  `crossover_op`) become **plain function references** (first-class functions —
  the operators are already pure functions, see §4.3); `None` means "algorithm
  default" exactly like torch.

**Construction policy — `make_<algorithm>` constructors (binding):** for every
algorithm config, the array-accepting entry point is a module-level functional
constructor `make_<algorithm>` defined in the SAME module as its config dataclass —
the workflow resolves the algorithm module via
`importlib.import_module(type(config).__module__)` and must find the constructor
there, next to `init`/`ask`/`tell`. The constructor is where ALL normalization and
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
leaves on the current etl master (commit b8062a9 "accept np.ndarray as static
trace values"), so the old rejection and its workarounds are obsolete — with ONE
exception: non-`None` callable fields are still rejected as static leaves anywhere
in the pytree. Zero-child `etl.register_pytree_node` registration is therefore
reserved for configs carrying callable op fields — `mo/nsga3`, `moead`, `rvea`
only. Every other config is a plain pytree and gets etl's by-value static
revalidation at `etl.run` (an opaque registered node is NOT revalidated).

**Carve-out — numerical problems (`problems/numerical/basic.py`,
`cec2022.py`):** these KEEP their validation-only `__post_init__` methods as the
sanctioned direct-construction public API — no ndarray→tuple normalization there
(`shift`/`affine` are stored as passed, validated only), and their clear
`ValueError` messages are asserted by the unit tests. The `make_*` pattern does
not apply to them.

### 4.2 State

- Each algorithm module defines a frozen dataclass (or namedtuple) `<Name>State`
  with a leaf per mutable quantity, e.g. `PSOState(population, velocity,
  local_best_location, local_best_fitness, global_best_location, global_best_fitness,
  key)`. All leaves are ETL tensors (float32 preferred for fitness/population;
  int32 for indices; keys are whatever `random.split` returns).
- `WorkflowState(algorithm_state, problem_state, monitor_state, generation, key)` —
  a dataclass in `core/workflow.py` (monitor_state may be `None` → use a constant
  dummy tensor leaf).
- Problem state: numerical problems are stateless → `ProblemState()` empty frozen
  dataclass (still a distinct type so the workflow can route it).
- Generation counter is an `int32` scalar tensor.
- Fitness dtype float32. Minimization semantics internally everywhere (mirror torch
  evox): the workflow applies `opt_direction` to convert maximization problems.

### 4.3 Functions (plain Python functions — NOT `@etl.defn`)

VERIFIED etl contract: `etl.trace`/`etl.build` accept plain callables; a `Defn`
object RAISES when called (even during another trace), so defn-wrapping would
break composition. Therefore **all algorithm/problem/monitor/operator functions
are PLAIN functions** (a bare call outside a trace raises etl's "No active
trace" TraceError — that is fine and expected). The workflow traces them
explicitly.

```python
# algorithms/<...>/<name>.py  — module-level functions + config dataclass
init(config, key) -> State                       # draw initial state
ask(config, state) -> (candidates, new_state)    # candidates: (n, dim) float32
tell(config, state, fitness) -> new_state        # fitness: (n,) or (n, n_obj) float32
# optional, when first-generation batch differs in size (NSGA-style):
init_ask(config, state) -> (candidates, state)
init_tell(config, state, fitness) -> state

# problems/<...>.py
evaluate(config, problem_state, pop) -> (fitness, problem_state)   # pop: (n, dim)
# numerical problems are stateless but keep the signature for uniformity.

# operators/<...>.py — pure functions (called inside traces)
# RULE (binding): keep the torch function name, argument names and order EXACTLY
# (torch operators are already pure functions in src/evox/operators/).
# - Functions that use randomness in torch (torch.rand/randint) gain `key` as
#   the FIRST parameter (use etl.random).
# - Drop the `device: torch.device` parameter (etl has no device arg).
# - `torch.Tensor` -> etl tensors; translate ops 1:1 (torch.where->etl.select,
#   torch.argsort->etl.argsort, torch.gather->etl.gather, clamp->clamp, etc.)
# Known torch signatures (from src/evox/operators/):
#   simulated_binary(x, pro_c, dis_c)                -> simulated_binary(key, x, pro_c, dis_c)
#   simulated_binary_half(x, pro_c, dis_c)           -> + key first
#   DE_differential_sum(diff_padding_num, num_diff_vects, index, population, F=None,
#                       replace=False)               -> + key first (F=None reproduces torch default)
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

**Static config args — no partialization needed (VERIFIED):** `etl.trace`/
`etl.build` treat each positional arg as one pytree: `TensorSpec` leaves become
tensor inputs, everything else (config dataclasses, ints, floats, strings)
becomes a STATIC value passed as-is into the function (static Python control
flow over them specializes the graph). So:

```python
exe = etl.build(algo_mod.ask, algo_config, state_spec_tree, backend=..., device=...)
state = exe.run(state)   # run() accepts/returns full pytrees
```

**Module convention (how the workflow finds the functions):** the `init`/`ask`/
`tell` (and `init_ask`/`init_tell`) functions live in the SAME module as their
config dataclass. The workflow resolves them via
`importlib.import_module(type(config).__module__)`. Same for problems
(`evaluate`) and monitors (`monitor_update`).

**Constants:** bake constant tensors inside functions with `etl.ops.constant`
(numpy array arg) — closure-captured CONCRETE tensors fail at trace time.
Python scalars in expressions are fine (constant broadcast).

**Key management convention** (JAX-evoX style): every `init/ask/tell` that needs
randomness splits the key FROM THE STATE: `key, subkey = random.split(state.key)`,
uses subkeys, and stores `key` back. Ask/tell must be deterministic given the state.

### 4.4 StdWorkflow (`core/workflow.py`, `workflows/std_workflow.py`)

Plain (non-frozen) class holding: algorithm config, problem config, monitor config
(or None), `opt_direction` ("min"/"max"/list), `num_generations`, backend name,
device, compile options. Public methods:

```python
wf = StdWorkflow(algorithm=..., problem=..., monitor=..., opt_direction="min", ...)
wf.init(seed=42)                      # builds init graph, runs once -> initial state
wf.step(state)                        # one step (compiled executable run)
wf.run(generations, seed=42) -> state # loop; collects monitor history
wf.fit(fitness=..., generations=...)  # mirror torch API if cheap
```

Implementation sketch (binding):
```python
step = _compose_step(...)      # module-level PLAIN function (see §4.3)
init_fn = algorithm_module.init
init_key_spec = etl.core.TensorSpec.from_tensor(key)
init_exe = etl.build(init_fn, algorithm_config, init_key_spec)   # cpu, numpy backend
state = init_exe.run(key)                                        # once
state = tree_map(lambda t: t.to(device), state)                  # if device is cuda
specs = etl.tree_map(etl.core.TensorSpec.from_tensor, state)
exe = etl.build(step, algorithm_config, problem_config, monitor_config, specs,
                backend=backend, device=device, **opts)          # compile ONCE
# per generation (same-device loop, pytrees in/out):
state = exe.run(state)
# host-side monitor history: after each step, copy out monitor state leaves
#    via leaf.to(etl.core.Device("cpu")).numpy() and append to Python lists.
```

The composed step (inside one graph): ask (or init_ask on gen==0) → evaluate →
opt-direction transform → tell (or init_tell) → monitor update → generation+1.
Use `etl.cond` for the first-generation branch. Every quantity stays in-graph;
Python-level history recording happens in the workflow loop AFTER each run.

### 4.5 Monitor (`core/monitor.py`, `workflows/eval_monitor.py`)

`MonitorState` dataclass holds tensor leaves (e.g. latest solution/fitness, elite
buffers of fixed size). `monitor_update(config, state, candidate, fitness) -> state`
is a `@etl.defn`. Host-side history lists (solution_history, fitness_history) live on
the monitor CONFIG object (plain Python lists) and are appended by the workflow
(post-step host copy). `EvalMonitor` must support both single- and multi-objective
(use `topk`/argsort for SO elite; non-dominated rank for MO Pareto front) and expose
`get_best_solution`/`get_best_fitness`/`plot` (plot can raise NotImplementedError if
plotly is absent — report).

## 5. Porting rules for algorithms

1. Port semantics 1:1 from the torch evox version in `src/evox/algorithms/` (read the
   file; the math must match). `bound_method` handling, batch processing, mirrored
   sampling etc. must be preserved.
2. Replace `torch.where` → `etl.select` (or `enp.where`), `torch.clamp` → `etl.clamp`,
   `torch.argsort` → `etl.argsort`, `torch.gather` → `etl.gather`, index writes →
   `etl.scatter`, `torch.cumsum` → `etl.cumsum`, matmul → `etl.dot`.
   torch `x.scatter`/index assignments must be rewritten with `etl.scatter` (replacement
   only — never accumulate; if the torch code scatter-ADDS, restructure to compute
   sums first, then scatter).
3. `torch.linalg.eigh` → `etl.eigh` (CMA-ES).
4. Any `torch.compile`/`vmap` annotations simply disappear — the whole step is one
   compiled graph.
5. RNG: replace `torch.rand(...)` etc. with `etl.random.*` + key splitting from
   `state.key`. Cross-check determinism: same seed → same trajectory (within fp32
   tolerance) across backends.
6. State dataclass field naming mirrors the torch evox `State` object's attribute names.
7. Configs are frozen dataclasses; if an algorithm truly needs mutable aux data
   (e.g. monitor history), it lives on the workflow/monitor config (host-side).
8. No Python control flow depending on tensor VALUES inside defn (use `etl.cond`/
   `select`). Static Python control flow on shapes/config is fine.
9. Numerical problems: `evaluate` = vectorized math over the (n, dim) tensor (never a
   Python loop over n). CEC2022: port the formulas AND the input-data loading
   (cec2022_input_data as numpy arrays loaded via importlib.resources; input data is
   CONSTANT tensors baked via closure, not graph inputs).
10. If some algorithm genuinely cannot be ported (uses an op etl lacks and no
    composition exists), DO NOT fake it: leave it out and report it to the
    parent agent with a reason. (Expectation: all 34 algorithms are portable.)

## 6. Testing strategy

`unit_test/etl/` mirrors the package:
- **Correctness**: port the torch evox unit tests (`unit_test/`) 1:1 where they are
  tensor-based; for every algorithm test, run the SAME problem/setup and assert the
  final best fitness is close to the torch evox result (tolerance 1e-3 relative for
  convergence-sensitive tests; for exact-determinism tests compare against etl numpy
  backend golden).
- **Cross-backend parity**: for each algorithm, run 3 generations with the numpy
  backend and (where applicable) assert outputs match within 1e-5/bit-exact.
- Run via the shared venv: `/mnt/local-ssd/bchuang/evox/.venv/bin/python -m pytest unit_test/etl -x -q`.
- IMPORTANT: test files must not import torch (etl-only tests). Parity tests that
  import both evox and evox_etl live in `unit_test/etl/parity/` and MAY import torch.

## 7. Benchmarks

`benchmarks/etl_vs_torch/` — comparison harness (see the CONTEXT.md there):
- SO: PSO, DE, CMA-ES, OpenES on Sphere/Rastrigin/Ackley; MO: NSGA2 (+ NSGA3, MOEAD)
  on DTLZ1/2.
- Scales: small (pop=100, dim=10), medium (pop=1000, dim=50), large (pop=10000,
  dim=100) — SO; (pop=100/1000, 3 obj, dim=10/30) — MO. 100 generations.
- Runners: torch-cpu, torch-cuda, etl-numpy-cpu, etl-iree-llvm-cpu, etl-iree-cuda,
  etl-xla-cuda. Same seeds/algorithms. Output: table of wall-time per step + total,
  fitness convergence, plus a markdown report comparing code style/cleanliness
  (pros/cons).

## 8. Environment

Shared venv: `/mnt/local-ssd/bchuang/evox/.venv/bin/python` (python3.11, torch
2.6.0+cu124 with working CUDA, evox + etl editable installs, jax-cuda12-pjrt plugin
available for etl-xla, pytest). ALWAYS use this interpreter for tests/benchmarks.
GPUs: 3× RTX A6000. Scan `nvidia-smi` for the most-free GPU before GPU runs.

## 9. ETL repo fixes (merged into etl master)

The xla-related etl fixes below are ALL merged into etl master (@f2f50a7) — the
shared venv's etl install needs no task branch:
- xla adapter GPU client-exhaustion fix (fresh PJRT client per compile/load →
  process SIGABRT; now a shared refcounted client — etl commit 838739c, formerly
  only on task branch `evogit-agent-T1-A19`);
- `GetPjRtApi` symbol-casing fallback + serialized `CompileOptionsProto` handling
  (etl commit 71d721e);
- the xla rank-0 staging fixes.
evox_etl code must NOT depend on any of these fixes — it only needs the etl API
already on master. If you hit an etl bug/missing feature while implementing,
record it in `src/evox_etl/CONTEXT.md` under "ETL issues found" (don't try to fix
etl yourself — escalate to the root agent).

## 10. Style rules

- Type hints on all public functions. Docstrings 1-3 lines.
- No classes with methods that mutate tensors. Functions only.
- Keep files < ~500 lines; mirror torch evox file names so diffing is easy.
- No torch/numpy imports in `src/evox_etl/**` (except `problems/numerical/cec2022.py`
  may use numpy ONLY to load the shipped input-data files; never in graph code).
- `from etl import tree_map, ...` and `import etl.numpy as enp`, `import etl.random as
  random` are the sanctioned imports. `etl` top level re-exports most ops.
