# evox_etl/core — protocols, state helpers, workflow

## Intent
The functional foundation of evox_etl: duck-typed protocol documentation (Algorithm:
`init/ask/tell` (+ optional `init_ask`/`init_tell`); Problem: `evaluate` (+ optional
`init`); Monitor: `monitor_update` (+ optional `init`)), state helpers
(`replace`/`get_nested`/`set_nested` + etl tree re-exports), and `StdWorkflow`
(compose+compile-once+run loop). See `../DESIGN.md` §4 — the binding spec.

## API Surface
- `state.py`: `replace(obj, **changes)` (dataclasses.replace, with a namedtuple
  `_replace` fallback), `get_nested(obj, path)` / `set_nested(obj, path, value)`
  (path = "a.b.c" or tuple of names), re-exports `tree_map/tree_leaves/
  tree_flatten/tree_unflatten` from etl.
- `algorithm.py` / `problem.py` / `monitor.py`: documentation-only `typing.Protocol`
  classes + type aliases (`AlgorithmState = Any`, ...). NO base classes required;
  functions are PLAIN module-level functions (NOT `@etl.defn`) living in the SAME
  module as their config dataclass; the workflow resolves them via
  `importlib.import_module(type(config).__module__)`.
- `workflow.py`: `EmptyState`, `WorkflowState(algorithm_state, problem_state,
  monitor_state, generation [0-d int32], key [0-d int64])` (both frozen dataclasses)
  and `StdWorkflow` (plain class).
- `StdWorkflow(algorithm, problem, monitor=None, opt_direction="min",
  solution_transform=None, fitness_transform=None, num_generations=None,
  backend="numpy", device=None, compile_options=None)`.
  Methods: `init(seed=42) -> WorkflowState` (builds init/step graphs once),
  `step(state=None) -> WorkflowState` (one `etl.run` of the compiled step exe +
  host-side monitor history recording), `run(generations=None, seed=42)`,
  `fit(fitness=None, generations=None, seed=42)` (needs a monitor wrapper with
  `get_best_fitness`). Internals: `opt_direction` property (tuple of ±1),
  `monitor` (host wrapper), `monitor_config` (completed cfg), `monitor_state`,
  `_step_exe`/`_init_exe` (cached executables).

## Constraints
- Everything runs on the default "numpy" backend, CPU only (no GPU in this dir's
  scope); use `/mnt/local-ssd/bchuang/evox/.venv/bin/python` for any run.
- NO eager etl ops — all etl ops only inside traces (workflow builds exes via
  `etl.build` and runs via `etl.run(exe, *args)`; `exe.run` does NOT exist).
- Step graph built ONCE: configs captured via CLOSURES (config dataclasses with
  callables/numpy arrays FAIL positional passing); the only graph inputs are state
  pytrees (+ key for init). Config fields = Python scalars/plain values.
- No torch imports; numpy allowed ONLY for baking constant arrays into graphs
  (`etl.ops.constant(etl.core.tensor(np.asarray(...)))` — inside the trace).
- State leaves are tensors ONLY; shapes must be static across runs (no
  dynamically-sized buffers in the compiled step graph — allocate full-size and
  track elites by value).
- Minimization semantics internally; workflow applies opt_direction (baked f32
  constant, scalar () or (m,)) BEFORE fitness_transform; monitor gets RAW
  candidates + TRANSFORMED fitness.

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
- `etl.cond(pred, true_fn, false_fn, *operands)` works with 0-d bool pred and
  dataclass pytree operands (used for the generation-0 branch).
- Reference for workflow behavior: `../../evox/workflows/std_workflow.py` (torch,
  read-only sibling). ETL same-device loop pattern validated in etl tests
  (`tests/backends/test_iree_same_device_loop.py` in the foreign repo).
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
- `global_best_fitness`-style fields are often shape (1,) — use
  `float(np.asarray(t.numpy()).reshape(-1)[0])` in tests, not `float(t.numpy())`.
