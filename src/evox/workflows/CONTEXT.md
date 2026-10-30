# evox/workflows + metrics + utils

## Intent
`workflows/` — `std_workflow.py` (StdWorkflow re-export, implementation in
`../core/workflow.py`) and `eval_monitor.py` (EvalMonitor: best-so-far tracking,
host-side history + Pareto front, algorithm auxiliary-tensor channel).
`metrics/` — igd.py, gd.py, hv.py (pure plain functions on evoxc tensors).
`utils/` — tree helpers, min_by, dominate_relation, pairwise_*_dist, cos_dist,
rank, rank_based_fitness, parse_opt_direction, cal_max.
See `../DESIGN.md`.

## workflows/ — implemented contract

- `std_workflow.py`: `from evox.core.workflow import EmptyState, StdWorkflow,
  WorkflowState` with `__all__`.
- `__init__.py`: exports `EvalMonitor`, `EvalMonitorConfig`, `StdWorkflow`,
  `WorkflowState`, `EmptyState` — `from evox.workflows import StdWorkflow,
  EvalMonitor` works.
- `eval_monitor.py` (function convention — config + `init`/`monitor_update` +
  wrapper class `Monitor = EvalMonitor` all in THIS module; the workflow resolves
  them via `type(config).__module__`):
  - `EvalMonitorConfig` frozen dataclass: `multi_obj, full_fit_history,
    full_sol_history, full_pop_history, topk, pop_size, dim, n_obj,
    opt_direction, fit_history, sol_history, pop_history` (lists host-side) plus
    `aux_history` (dict host-side, LAST field so the order is back-compatible).
    `aux_history` maps `str` key -> `list[np.ndarray]` (one entry per generation).
    The workflow completes `pop_size`/`dim` (from algorithm state
    `population`/`pop` or config), `n_obj`/`multi_obj` (from a list
    opt_direction), and always overwrites `opt_direction` with its own
    tuple-of-±1 (workflow direction is authoritative).
  - `init(config, key)` (key unused): SO — `EvalMonitorState(latest_solution
    (pop,dim) zeros, latest_fitness (pop,) zeros, topk_solutions (topk,dim)
    zeros, topk_fitness (topk,) +inf)`; MO — `MOEvalMonitorState(latest_solution,
    latest_fitness (pop,n_obj) zeros)` ONLY. Zeros/+inf are in-graph
    `evoxc.ops.constant(evoxc.core.tensor(np...))` (no eager `evoxc.zeros`, no
    `*_like`). Raises ValueError if pop_size/dim (or n_obj in MO) are None.
    +inf topk init makes the running elite monotone (gen-0 always wins).
  - `monitor_update(config, state, candidate, fitness)` branches on STATIC
    `len(fitness.shape)`: rank-1 (SO) → concat stored elite with the new gen,
    `evoxc.topk(concat_fit, config.topk, axis=0, largest=False)` (returns
    (values, int64 indices) tuple), row-gather solutions via
    `evoxc.gather(concat_sol, indices, axis=0)` (numpy-take semantics); rank-2
    (MO) → only latest_solution/latest_fitness.
  - SHAPE POLICY (the monitor handles ANY leading dim):
    `monitor_update` stores the FULL incoming batch as-is, so
    `latest_solution`/`latest_fitness` take the shape of whatever batch the
    algorithm's step last passed to `evaluate`, NOT necessarily
    `(pop_size, ...)` — the raw evaluated batch is stored, never sliced to
    pop_size. CoDE evaluates `(3*pop_size, dim)` per
    generation; CSO evaluates `pop_size // 2` after its init_step. The SO
    top-k pools the elite with the ENTIRE evaluated batch (for CoDE the
    best-of-3n trials can win), and history entries are the full per-gen
    batches (CoDE `(3n,)` fitness; CSO `(n,)` at gen 0 then `(n/2,)`). The
    `(pop_size, dim)` zeros / `(topk,)` +inf buffers from `init` are gen-0
    placeholders only — every leaf is overwritten on the first
    `monitor_update` (evoxc states must be full-size tensors from init; there is
    no unallocated placeholder buffer).
    CONSEQUENCE: monitor-state leaf shapes may CHANGE between generations, so
    a compiled step graph is valid for exactly one state-shape signature —
    `StdWorkflow` keys its per-variant exe cache by
    `(variant, leaf-shape/dtype signature)` and re-traces on drift (fix for
    the CoDE/CSO gen-2 ShapeError; see `../core/workflow.py`
    `_state_signature`).
  - `record_auxiliary(config, aux)` — module-level PLAIN (non-defn) HOST-side
    function, same module as the config. Gated on `config.full_pop_history`
    (returns immediately when off). For each `(key, value)` in the
    `aux` dict, the value is a CONCRETE evoxc tensor converted to numpy via
    `.to(evoxc.core.Device("cpu")).numpy()` (the `EvalMonitor._to_numpy` idiom) and
    appended to `config.aux_history.setdefault(key, [])` (list created on first
    use). Source of the algorithm auxiliary-tensor channel; the workflow feeds it
    each generation.
  - Wrapper `EvalMonitor(config=..., state=...)` — workflow constructs it and
    reassigns `.state` after every step. All accessors return numpy (concrete
    evoxc leaves via `t.to(evoxc.core.Device("cpu")).numpy()`); stored fitness is
    opt-direction-scaled, so `get_latest_fitness`/`get_topk_fitness`/
    `get_best_fitness`/`get_fitness_history`/`get_pf_fitness` multiply by
    `config.opt_direction` (a `(1,)` direction acts as a scalar; `get_best_fitness`
    returns a Python float). SO-only accessors (`get_topk_*`, `get_best_*`) raise
    the documented ValueError under MO; MO-only `get_pf*` raise under SO.
    `get_pf_fitness` dedupes on FITNESS (np.unique axis=0), `get_pf`/
    `get_pf_solutions` on SOLUTIONS (the two can differ).
    Non-domination ranks are computed host-side via a lazily cached exe:
    `evoxc.build(non_dominate_rank, TensorSpec(shape, np.float32), backend="numpy",
    device=Device("cpu"))` + `evoxc.run(exe, evoxc.core.tensor(arr))`, cached per
    input shape. `fitness_history`/`fit_history`, `solution_history`/
    `sol_history` properties alias the config lists. `aux_history` /
    `auxiliary_history` (alias) return `config.aux_history`. `pop_history`
    returns `config.aux_history["pop"]` when that key is present, else falls back
    to the legacy `config.pop_history` list (backwards compat); the workflow's
    own `pop_history` recording still appends to `config.pop_history`.
    `record_auxiliary(self, aux)` delegates to the module-level
    `record_auxiliary(self.config, aux)`.
    `plot(problem_pf=None, source="eval", **kwargs)` is a plain host-side method
    returning a `plotly.graph_objects.Figure` (no torch imports).
    Warns + returns None when nothing was recorded
    (`not self.fitness_history and not self.aux_history` →
    `"No fitness history recorded, return None"`) or when the visualization tool
    is unavailable (`_vis_plot_module()` probe: `evox.vis_tools` imports fine
    WITHOUT plotly, so availability is `evox.vis_tools.plot.go is not None` →
    `'No visualization tool available, return None. Hint:
    pip install "evox[vis]"'`). Source dispatch: `"pop"` → the RAW
    `self.aux_history["fit"]` channel (NOT un-negated), `"eval"` →
    `self.get_fitness_history()` (un-negated), else ValueError. History entries
    are `np.asarray`-ed, `n_objs` is `1` when `fitness_history[0].ndim == 1` else
    `self.fitness_history[0].shape[1]`, and the figure comes from
    `vis_tools.plot_obj_space_{1d,2d,3d}` (`**kwargs` forwarded; 2d/3d receive
    `problem_pf` positionally); ≥4 objectives warn `"Not supported yet."` +
    return None.
- Workflow tweaks in `../core/workflow.py`: `_discover_pop_size` falls back to
  the `pop` state attribute when `population` is absent (PSO/NSGA2 states);
  `_complete_monitor_config` fills `opt_direction` from the workflow.
- History appends are PIPELINED in `../core/workflow.py` (`_readback` /
  `_drain_readbacks`): `fit_history`/`sol_history`/`pop_history` entries are
  copied device→host with `evoxc.io.sink` at the end of generation k and appended
  at the start of generation k+1, so the copy overlaps the next step graph.
  Values, order and the final lists are unchanged; `run`/`fit` flush the last
  generation themselves, a caller driving `step()` directly must call
  `StdWorkflow.flush()` before reading the history mid-run. The
  `record_auxiliary` channel (`config.aux_history`) stays eager. Cost model and
  measured numbers: `../core/CONTEXT.md` → "Monitor readback pipelining".
- Validation (venv interpreter, `sys.path` + `src/`): PSO+Sphere+min converges
  monotone to ~0.028 @ gen 50 (6.3 → 4.0 between gen 30 and 40),
  `fit()` returns a scalar float, opt_direction="max" un-negates correctly;
  NSGA2+DTLZ2(m=3)+`["min"]*3` after 3 steps gives `get_pf_fitness()` (k, 3).
  Note: `get_pf*` with history flags off warns then fails on empty history
  (concatenating nothing is an error by construction).
- Validation of the shape policy (throwaway drivers, numpy backend):
  CoDE(20,[-5,5]^4)+Sphere+min, 20 gens → best ≈ 9.9e-3, `fit_history` entries
  all `(60,)`, `sol_history` `(60, 4)` (full 3n batch),
  `get_topk_fitness()[0] == get_best_fitness()`; CSO(20)+Sphere, 30 gens →
  best ≈ 5.2 (slow but converging), history `(20,)` then `(10,)` per gen;
  CSO + opt_direction="max" → `get_best_fitness()` ≈ 80.9 (>0, un-negated);
  NSGA2+DTLZ2 pf_fitness `(k, 3)` finite; PSO(40)×50 gens unchanged (best
  3.23e-07, deterministic under equal seeds).
- Validation of the aux channel (throwaway driver, numpy backend): with
  `full_pop_history=True` three `record_auxiliary(cfg, {"center","pop","fit"})`
  calls give `cfg.aux_history` keys with 3 numpy entries each of the source
  shapes (`(2,)`, `(4, 2)`, `(4,)`); wrapper `aux_history`/`auxiliary_history`
  are `cfg.aux_history` and `pop_history` is `aux_history["pop"]`; with
  `full_pop_history=False` nothing is recorded; without a `"pop"` aux key
  `pop_history` returns `cfg.pop_history`; fresh config defaults are an empty
  dict / empty list.

## Deliberate scope limits (vs the 1.x torch line, removed in 2.0)
Current-state inventory of what the 1.x torch API had and this node does not:
- No distributed multi-rank evaluation: the 1.x `StdWorkflow(enable_distributed,
  group)` split one population across ranks, gathered fitness and forked the RNG;
  this workflow runs one device per workflow (multi-GPU = one process per GPU
  only). Documented user-facing in
  `docs/source/guide/experimental/distributed_workflow.md`.
- Monitor hooks are collapsed to ONE `monitor_update(config, state, candidate,
  fitness)`; the 1.x `Monitor.pre_eval` (post-solution-transform) and `post_eval`
  (pre-fitness_transform) observations no longer exist, and monitor code runs
  IN-GRAPH, so arbitrary host-side side effects per hook are gone (host-side
  effects only via the algorithm `record_step` -> monitor `record_auxiliary` aux
  channel).
- No `set_config(**config)`, no user-constructed/shared monitor instance, no
  `get_submodule("monitor")`: the workflow owns the wrapper (`wf.monitor` is None
  until `init`), and one `EvalMonitorConfig` must not be shared across workflows
  (its history lists would interleave).
- No `device`/`history_device` knobs on the monitor: history is always host numpy
  (D2H copies pipelined by `_readback`), never GPU-resident.
- `EvalMonitor.get_best_fitness()` returns a Python `float` (1.x returned a 0-d
  tensor), and every accessor returns numpy — the `monitor.get_best_fitness()
  .item()` idiom breaks.
- MO detection is static: `multi_obj`/`n_obj` are filled only from a LIST
  `opt_direction`; there is no runtime rank-2 fitness sniffing.
- Aux/pop history coverage: only 13 of 32 algorithm modules define the
  `record_step` hook (the ES variants). For the other 19, `aux_history` stays
  empty, so `plot(source="pop")` raises KeyError, and the legacy `pop_history`
  fallback records the last EVALUATED batch rather than the post-step population.
- History is appended once per workflow step, not once per `evaluate` call, so an
  algorithm calling `evaluate` several times per generation yields fewer entries.
- No batched/vmapped workflows (`use_state` + `vmap` + `stack_module_state` in
  1.x); `evoxc.vmap` exists but is not wired into `StdWorkflow`.
- No `nn.Module` interop: no `state_dict`/`load_state_dict`/`parameters()`/
  `get_submodule`/`train`-`eval`/`torch.save(workflow)`; `solution_transform`/
  `fitness_transform` must be evoxc-traceable plain callables.
- No public workflow `.state` accessor for reading algorithm state after a run
  (the returned `WorkflowState`/the private `wf._state` are the only handles).
- Validation errors are `TypeError`/`ValueError` (1.x raised `AssertionError`,
  e.g. for a bad `opt_direction`).
