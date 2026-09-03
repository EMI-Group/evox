# evox_etl/workflows + metrics + utils

## Intent
`workflows/` — `std_workflow.py` (StdWorkflow re-export, implementation in
`../core/workflow.py`) and `eval_monitor.py` (EvalMonitor: best-so-far tracking,
host-side history + Pareto front).
`metrics/` — igd.py, gd.py, hv.py (pure defn functions; mirror torch evox math in
`../../evox/metrics/`, read-only).
`utils/` — tree helpers, min_by, dominate_relation, pairwise_*_dist, cos_dist,
rank, rank_based_fitness, parse_opt_direction, cal_max (mirror `../../evox/utils/`).
See `../DESIGN.md`.

## workflows/ — implemented contract

- `std_workflow.py`: `from evox_etl.core.workflow import EmptyState, StdWorkflow,
  WorkflowState` with `__all__`.
- `__init__.py`: exports `EvalMonitor`, `EvalMonitorConfig`, `StdWorkflow`,
  `WorkflowState`, `EmptyState` — `from evox_etl.workflows import StdWorkflow,
  EvalMonitor` works.
- `eval_monitor.py` (function convention — config + `init`/`monitor_update` +
  wrapper class `Monitor = EvalMonitor` all in THIS module; the workflow resolves
  them via `type(config).__module__`):
  - `EvalMonitorConfig` frozen dataclass: `multi_obj, full_fit_history,
    full_sol_history, full_pop_history, topk, pop_size, dim, n_obj,
    opt_direction, fit_history, sol_history, pop_history` (lists host-side).
    The workflow completes `pop_size`/`dim` (from algorithm state
    `population`/`pop` or config), `n_obj`/`multi_obj` (from a list
    opt_direction), and always overwrites `opt_direction` with its own
    tuple-of-±1 (workflow direction is authoritative).
  - `init(config, key)` (key unused): SO — `EvalMonitorState(latest_solution
    (pop,dim) zeros, latest_fitness (pop,) zeros, topk_solutions (topk,dim)
    zeros, topk_fitness (topk,) +inf)`; MO — `MOEvalMonitorState(latest_solution,
    latest_fitness (pop,n_obj) zeros)` ONLY. Zeros/+inf are in-graph
    `etl.ops.constant(etl.core.tensor(np...))` (no eager `etl.zeros`, no
    `*_like`). Raises ValueError if pop_size/dim (or n_obj in MO) are None.
    +inf topk init makes the running elite monotone (gen-0 always wins).
  - `monitor_update(config, state, candidate, fitness)` branches on STATIC
    `len(fitness.shape)`: rank-1 (SO) → concat stored elite with the new gen,
    `etl.topk(concat_fit, config.topk, axis=0, largest=False)` (returns
    (values, int64 indices) tuple), row-gather solutions via
    `etl.gather(concat_sol, indices, axis=0)` (numpy-take semantics); rank-2
    (MO) → only latest_solution/latest_fitness.
  - Wrapper `EvalMonitor(config=..., state=...)` — workflow constructs it and
    reassigns `.state` after every step. All accessors return numpy (concrete
    etl leaves via `t.to(etl.core.Device("cpu")).numpy()`); stored fitness is
    opt-direction-scaled, so `get_latest_fitness`/`get_topk_fitness`/
    `get_best_fitness`/`get_fitness_history`/`get_pf_fitness` multiply by
    `config.opt_direction` (a `(1,)` direction acts as a scalar; `get_best_fitness`
    returns a Python float). SO-only accessors (`get_topk_*`, `get_best_*`) raise
    the torch ValueError under MO; MO-only `get_pf*` raise under SO.
    `get_pf_fitness` dedupes on FITNESS (np.unique axis=0), `get_pf`/
    `get_pf_solutions` on SOLUTIONS (torch semantics — the two can differ).
    Non-domination ranks are computed host-side via a lazily cached exe:
    `etl.build(non_dominate_rank, TensorSpec(shape, np.float32), backend="numpy",
    device=Device("cpu"))` + `etl.run(exe, etl.core.tensor(arr))`, cached per
    input shape. `fitness_history`/`fit_history`, `solution_history`/
    `sol_history`, `pop_history` properties alias the config lists.
    `plot()` raises NotImplementedError (with the plotly-install message) when
    plotly is missing, otherwise NotImplementedError for the plotting itself;
    warns + returns None when no history is recorded.
- Workflow tweaks in `../core/workflow.py`: `_discover_pop_size` falls back to
  the `pop` state attribute when `population` is absent (PSO/NSGA2 states);
  `_complete_monitor_config` fills `opt_direction` from the workflow.
- Validation (venv interpreter, `sys.path` + `src/`): PSO+Sphere+min converges
  monotone to ~0.028 @ gen 50 (torch ref 0.012; 6.3 vs 4.0 @ gen 30 — parity),
  `fit()` returns a scalar float, opt_direction="max" un-negates correctly;
  NSGA2+DTLZ2(m=3)+`["min"]*3` after 3 steps gives `get_pf_fitness()` (k, 3).
  Note: `get_pf*` with history flags off warns then fails on empty history
  (torch-parity: `torch.cat([])` fails the same way).
