# evox_etl/core — protocols, state helpers, workflow

## Intent
The functional foundation of evox_etl: duck-typed protocol documentation (Algorithm:
STEP protocol — `init`/`step` + optional `init_step`/`final_step`/`record_step`/
`monitor_candidate`, with a
workflow-injected `evaluate` closure; Problem: `evaluate` (+ optional `init`); Monitor:
`monitor_update` (+ optional `init`/`record_auxiliary`)), state helpers
(`replace`/`get_nested`/`set_nested` + etl tree
re-exports), and `StdWorkflow` (compose+compile-once-per-variant+run loop). See
`../DESIGN.md` §4 — the binding spec.

## API Surface
- `state.py`: `replace(obj, **changes)` (dataclasses.replace, with a namedtuple
  `_replace` fallback), `get_nested(obj, path)` / `set_nested(obj, path, value)`
  (path = "a.b.c" or tuple of names), re-exports `tree_map/tree_leaves/
  tree_flatten/tree_unflatten` from etl.
- `algorithm.py` / `problem.py` / `monitor.py`: documentation-only `typing.Protocol`
  classes + type aliases (`AlgorithmState = Any`, ..., `Evaluate = Callable[[Candidates],
  Fitness]`). NO base classes required; functions are PLAIN module-level functions
  (NOT `@etl.defn`) living in the SAME module as their config dataclass; the workflow
  resolves them via `importlib.import_module(type(config).__module__)`.
- Optional auxiliary-history hooks (documented in the Protocols, detected generically
  via `getattr`, absent from all current algorithm/monitor modules): algorithm
  `record_step(config, state, candidate, fitness) -> dict[str, tensor]` and monitor
  `record_auxiliary(config, aux) -> None` — both PLAIN module-level HOST-SIDE functions
  (NOT `@etl.defn`, never called inside a trace; no `__all__` entry, they are Protocol
  methods).
- Optional monitor-candidate hook (also detected via `getattr`, called INSIDE the
  trace): algorithm `monitor_candidate(candidates) -> tensor`. Default = identity,
  so a `(pop_size, dim)` tensor candidate reaches `monitor_update` unchanged; it
  exists for algorithms that hand `evaluate` a non-tensor payload (the virtual
  ES family passes a `(center, seeds, sigma)` tuple — its hook broadcasts the
  `(dim,)` center to the `(pop_size, dim)` the monitor concatenates). Only invoked
  when a monitor is configured.
- `workflow.py`: `EmptyState`, `WorkflowState(algorithm_state, problem_state,
  monitor_state, generation [0-d int32], key [0-d int64])` (both frozen dataclasses)
  and `StdWorkflow` (plain class).
- `StdWorkflow(algorithm, problem, monitor=None, opt_direction="min",
  solution_transform=None, fitness_transform=None, num_generations=None,
  backend="numpy", device=None, compile_options=None)`. The algorithm module MUST
  define a callable `step(config, state, evaluate)` (TypeError at construction
  otherwise); `init_step`/`final_step` are optional. Methods: `init(seed=42) ->
  WorkflowState` (builds init graph + pre-builds the `step` variant graph for the
  initial signature), `init_step(state=None)` / `step(state=None)` /
  `final_step(state=None)` (public step API; each runs ONE generation, records
  monitor history, updates `self._state`, returns the new state), `run(generations=
  None, seed=42)`, `fit(fitness=None, generations=None, seed=42)` (needs a monitor
  wrapper with `get_best_fitness`). Internals: `opt_direction` property (tuple of ±1),
  `monitor` (host wrapper), `monitor_config` (completed cfg), `monitor_state`,
  `_has_init_step`/`_has_final_step`/`_has_record_step` (host-side callable-presence
  flags, resolved once in `__init__` via `getattr`),
  `_step_exes` (dict `(resolved variant, state signature)` → exe, lazily built),
  `_state_signature(state)` (staticmethod: ordered `(shape, dtype)` tuple of all
  tensor leaves — the `_spec_key` idiom from
  `unit_test/etl/algorithms/helpers.py`), `_init_exe`/`_mon_init_exe`.

## The step protocol (BINDING for algorithm modules)
- `init(config, key) -> state` — unchanged from the pre-1.0 protocol.
- `step(config, state, evaluate) -> state` — REQUIRED. Owns ONE full generation:
  produce candidates (old ask-body), get fitness via `fitness = evaluate(candidates)`
  (may call multiple times with different candidate sets), update state (old
  tell-body), return state. The workflow handles the generation counter and the key
  passthrough — the returned state's `key` management is the algorithm's own concern
  (split-from-state convention unchanged).
- `init_step(config, state, evaluate) -> state` — OPTIONAL first-generation variant.
  Absent → the workflow's `init_step()` calls the module's `step`.
- `final_step(config, state, evaluate) -> state` — OPTIONAL last-generation variant.
  Absent → `final_step()` calls `step`.
- `ask`/`tell`/`init_ask`/`init_tell` are DELETED from the protocol.
- `evaluate(candidates) -> fitness` is a traced closure created by the WORKFLOW
  inside the step graph. Pipeline per call: solution_transform → `problem.evaluate`
  (threads problem state) → opt-direction scaling (min semantics, baked f32
  constant, scalar () or (m,)) → fitness_transform → `monitor_update` (threads
  monitor state; receives RAW candidates + TRANSFORMED fitness — the torch
  post_ask/pre_tell semantics). Algorithms treat it as fully opaque: pass whatever
  tensor/pytree the candidates are, get transformed fitness back; do NOT store or
  re-thread it. The workflow keeps the problem/monitor state of the LAST evaluate
  call (torch's stateful Problem semantics, reified).
- Dispatch is HOST-SIDE: `StdWorkflow.__init__` records `_has_init_step`/
  `_has_final_step` = whether the module defines callables; `init_step()`/`
  final_step()` resolve to the module function if present else `step`. There is NO
  in-graph `generation == 0` branch anymore.
- One compiled exe per (RESOLVED variant, STATE SIGNATURE)
  (`_step_exes: dict[tuple[str, tuple], Executable]`), built lazily from the
  CURRENT state's TensorSpec tree on a cache miss; the `step` variant is
  pre-built in `init()` for the initial signature. A step-only algorithm with a
  stable state therefore reuses a single exe, but state leaves legitimately
  RESIZE between generations (CoDE hands the monitor `(3·pop_size, dim)`
  candidate batches vs `(pop_size, dim)` init placeholders; CSO stores
  `(pop_size/2, dim)` after its init_step), so a drifted signature triggers a
  fresh trace keyed under the new signature — no ShapeError at generation 2.
  EXTRA exes are only built for signatures that actually occur; the signature
  stabilizes after the first drift (per algorithm/problem shapes, one retrace
  per variant).
- `run(generations=N)`: `init` + `init_step` + (N−2) plain steps + (final_step if
  module defines it else step) — i.e. the LAST generation uses final_step; `N == 1`
  runs only init_step; total generations including init_step == N. `fit` uses the
  same loop (plus the generation-0 `fit_history` append when `fitness` is given)
  and returns `monitor.get_best_fitness()`.

## Auxiliary-history channel (HOST-SIDE, optional)
- The algorithm module MAY define `record_step(config, state, candidate, fitness) ->
  dict[str, etl_tensor]` (see `algorithm.py`); the monitor module MAY define
  `record_auxiliary(config, aux: dict) -> None` (see `monitor.py`).
- Both are PLAIN module-level host-side functions in the same module as their config
  (NOT `@etl.defn`); both are detected generically via `getattr`, never required.
- Flow, once per generation, HOST-SIDE after each `_run_variant` (`_record_history`
  calls it, so it is NOT part of any traced graph): `record_step(config, post_step_state,
  candidate, fitness)` → `record_auxiliary(monitor_config, aux)` → the monitor stores
  it as host-side aux history (EvalMonitor: `config.aux_history`).
- `candidate`/`fitness` are the CONCRETE etl tensors `state.monitor_state.latest_solution`
  / `.latest_fitness` (or None if the monitor state lacks them); `state` is the
  POST-step algorithm state; a None return from `record_step` skips `record_auxiliary`.
- The aux block runs whenever `self.monitor_config is not None`, BEFORE the
  fit/sol/pop `has_history` early-return, so a monitor config with none of those
  field names still receives aux history.
- `pop_history` fallback rule: the legacy `latest_solution` append into
  `config.pop_history` happens ONLY when `_has_record_step` is False; when a
  `record_step` hook exists the monitor derives `pop_history` from `aux_history["pop"]`,
  so the workflow does not double-record. `fit_history`/`sol_history` appends are
  unaffected (still gated on their `full_*_history` flags).

## Constraints
- Everything runs on the configured backend/device; the default is the "numpy"
  backend on cpu; use the documented venv for any run.
- NO eager etl ops — all etl ops only inside traces (workflow builds exes via
  `etl.build` and runs via `etl.run(exe, *args)`; `exe.run` does NOT exist).
- Step graphs built ONCE per variant: configs captured via CLOSURES (config
  dataclasses with callables/numpy arrays FAIL positional passing); the only graph
  inputs are state pytrees (+ key for init). Config fields = Python scalars/plain
  values.
- No torch imports; numpy allowed ONLY for baking constant arrays into graphs
  (`etl.ops.constant(etl.core.tensor(np.asarray(...)))` — inside the trace).
- State leaves are tensors ONLY; shapes must be static across runs (no
  dynamically-sized buffers in the compiled step graphs — allocate full-size and
  track elites by value).
- Minimization semantics internally; workflow applies opt_direction BEFORE
  fitness_transform; monitor gets RAW candidates + TRANSFORMED fitness.

## Notes for Agents (verified ETL gotchas)
- `etl.sum`/`mean`/reductions take `axes=`, but `topk`/`argmin`/`gather` take
  `axis=` — inconsistent keyword naming.
- Dynamic indexing `x[idx]` with a symbolic scalar tensor is NOT supported
  ("getitem: unsupported index key SymbolicTensor") — use `etl.gather(x, idx, axis=0)`.
- `etl.select(pred, a, b)` broadcasts pred with a/b directly (unlike torch.where's
  leading-dim broadcast): reshape the pred, e.g. `etl.reshape(cond, (n, 1))`.
- `etl.tree_unflatten(leaves, treespec)` — argument order REVERSED vs JAX.
- int32 tensor + Python int → int64; int32 * Python float → float64 (f32 + Python
  float stays f32). Cast generation counters explicitly to np.int32.
- `etl.ops.constant` (and every op) fails OUTSIDE a trace (TraceError); constant()
  rejects raw ndarray — wrap via `etl.core.tensor(...)` first, inside the trace.
- Top-level `etl.zeros`/`etl.ones`/`etl.full` are EAGER constructors returning
  CONCRETE `Tensor`s — NOT usable inside traces (graph outputs must be
  `SymbolicTensor`s). For in-graph constants always use
  `etl.ops.constant(etl.core.tensor(np.asarray(...)))`.
- `TensorSpec(shape_tuple, np_dtype)`; build via
  `etl.tree_map(lambda t: etl.core.TensorSpec(tuple(t.shape), t.dtype), state)`
  (no TensorSpec.from_tensor). Empty dataclasses (e.g. `EmptyState`) flow through
  build/run as empty pytree containers.
- `dataclasses.replace` does NOT work on namedtuples (py3.11) — use `state.replace`.
- Eager (outside trace) is fine: `etl.random.key(seed)`, `t.shape`/`t.dtype`/
  `t.numpy()`, `t.to(etl.core.Device("cpu"))`, `etl.core.tensor(np_array)`.
- NONLOCAL CLOSURES INSIDE TRACES WORK (validated): a traced step body can define
  `evaluate` with `nonlocal prob_state, mon_state` rebinding across multiple calls —
  this is exactly how the workflow threads problem/monitor state inside one
  generation. etl traces plain Python closures, not just pure expressions.
- MODULE CONVENTION GOTCHA: the workflow resolves functions via
  `type(config).__module__`, so algorithm/problem/monitor configs used together
  MUST live in DISTINCT modules — two configs defined in the same module (e.g.
  both inline in `__main__` or in one test file) collide on function names like
  `init`. Tests must define toy components in separate module files.
- TEST SEMANTICS: monitor `fit_history` entries are per-generation latest-fitness
  VECTORS (torch EvalMonitor semantics — NOT monotone). Assert convergence via the
  monitor's running best (topk_fitness / get_best_fitness) or the algorithm's
  internal best field — those are monotone. With opt_direction="max": internal
  (minimized) fitness decreases while get_best_fitness (un-negated) increases.
  NOTE: the monitor config's `*_history` LISTS ACCUMULATE across `run()` calls on
  the same workflow instance (pre-existing EvalMonitor semantics).
- `global_best_fitness`-style fields are often shape (1,) — use
  `float(np.asarray(t.numpy()).reshape(-1)[0])` in tests, not `float(t.numpy())`.
- Reference for workflow behavior: `../../evox/workflows/std_workflow.py` and
  `../../evox/core/components.py` (torch, read-only siblings). ETL same-device loop
  pattern validated in etl tests (`tests/backends/test_iree_same_device_loop.py`
  in the foreign repo).
