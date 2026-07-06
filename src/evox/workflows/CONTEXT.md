# Workflows (`src/evox/workflows`)

## Intent
The workflows module provides the **orchestration layer** that connects Algorithms to Problems in EvoX. It houses the standard evolutionary optimization loop and its companion evaluation monitor.

## API Surface

| Export | Source | Role |
|---|---|---|
| `StdWorkflow` | `std_workflow.py` | Standard optimization workflow — wires together Algorithm + Problem + Monitor |
| `EvalMonitor` | `eval_monitor.py` | Evaluation monitor — records fitness/solution history, tracks elites, computes Pareto fronts |

## Design

### `StdWorkflow` — The Optimization Loop

`StdWorkflow` extends `evox.core.Workflow`. It orchestrates a single evolutionary step through this lifecycle:

```
Algorithm asks for population
   → Monitor.post_ask(pop)
   → solution_transform(pop)         [user-provided nn.Module]
   → Monitor.pre_eval(transformed)
   → Problem.evaluate(transformed)   [optionally distributed across GPUs]
   → Monitor.post_eval(fitness)
   → fitness_transform(fitness)      [user-provided nn.Module, prepended with OptDirectionTransform]
   → Monitor.pre_tell(transformed_fitness)
   → returns fitness to Algorithm
```

Key behaviors:
- **`opt_direction`**: Accepts `"min"` (default) or `"max"` (or a list for per-objective directions). Internally, EvoX always minimizes; `"max"` negates fitness via `OptDirectionTransform` before the user's `fitness_transform` and the monitor.
- **Algorithm wrapping**: The constructor wraps the user's algorithm in a dynamic subclass that overrides `evaluate()`, redirecting it through `StdWorkflow._evaluate()` — this is how the monitor hooks and transforms are injected.
- **Distributed**: When `enable_distributed=True`, the population is split across the process group, each rank evaluates its chunk (with forked RNG), and results are gathered via `all_gather`.
- **Lifecycle**: Respects optional `init_step()` / `final_step()` on the algorithm (detected by comparing method identity to the base `Algorithm` class).

### `EvalMonitor` — Tracking & Elite Management

`EvalMonitor` extends `evox.core.Monitor`. It hooks around the evaluation step to:

- Track `latest_solution`, `latest_fitness` (via `Mutable` tensors live on the device)
- Maintain top‑k elite solutions/fitness for **single‑objective** optimization
- Record full history of fitness and/or solutions into a module-level `__monitor_history__` dict (keyed by monitor instance id, with weakref cleanup)
- Compute **Pareto fronts** for multi‑objective optimization via `non_dominate_rank`
- Record auxiliary (population) history from the algorithm's `record_step()` output
- Support `torch.compile` via a `_data_sink` token‑passing pattern with dummy/vmap variants

Query API:
- Single‑objective: `get_best_solution()`, `get_best_fitness()`, `get_topk_solutions()`, `get_topk_fitness()`
- Multi‑objective: `get_pf()`, `get_pf_fitness()`, `get_pf_solutions()`
- History: `get_fitness_history()`, `get_solution_history()`, `fitness_history`, `solution_history`, `auxiliary_history`
- Visualization: `plot()` (requires optional `evox.vis_tools`)

**opt_direction handling**: The monitor stores `opt_direction` and uses it to negate raw fitness in accessor methods (e.g., `get_best_fitness()` returns original‑scale values), while internal storage uses minimized values.

## Constraints
- `solution_transform` and `fitness_transform` must be `torch.compile`‑compatible (nn.Module or callable)
- When using `enable_distributed`, the problem evaluation must be RNG‑independent across individuals (handled by `fork_rng`)
- `EvalMonitor` with `full_sol_history=True` can consume significant memory — use `history_device="cpu"` (default) to avoid GPU memory pressure
- `Monitor` is optional in `StdWorkflow`; a plain `Monitor()` base is used if none is provided

## Routing Table
- `std_workflow.py` → Standard optimization workflow (`StdWorkflow`)
- `eval_monitor.py` → Evaluation monitor with history tracking (`EvalMonitor`)
