# Hyperparameter optimization (HPO)

Hyperparameter optimization tunes the *settings* of an evolutionary run — the inertia weight of PSO, the crossover rate of a DE variant, a population size, a monitor option — rather than the decision variables of the problem. In `evox` this is done **host-side**: a `HPOProblemWrapper` runs a complete inner `StdWorkflow` once per candidate configuration and returns the inner run's best fitness, and a host-side search loop (`random_search`, or any external optimizer) drives the wrapper.

```{warning}
This page documents the **experimental EvoX 2.0 pre-release**. HPO in EvoX 2.0 is **host-side**, not a `Problem` you hand to another `StdWorkflow`. The 1.x design — wrapping a whole workflow as an outer `Problem` and optimizing its hyperparameters *inside* the outer workflow's traced `evaluate` — has no `evox` counterpart and is not coming back; see below for why. The host-side API may still change before 2.0 final.
```

## Why host-side

This is the single most important point on the page. `evoxc` (the tensor-graph compiler behind `evox`, see {doc}`backends`) has **no eager mode**: every tensor operation needs an active trace, and `evoxc.run` — the only way to execute anything — **cannot be called while a trace is active**. An inner `StdWorkflow` step is a compiled `Executable`, so an `evaluate(candidates)` closure traced by an *outer* `StdWorkflow` could never call it: evaluating one hyperparameter candidate would mean submitting a nested graph from inside a trace. `evoxc` also has no analogue of `torch.func.stack_module_state` (or of the `register_vmap` custom-op loop the torch version used to map the inner state over the outer population), so candidates cannot even be batched in-graph. A faithful in-graph HPO is therefore not portable, and `HPOProblemWrapper` does not pretend otherwise:

- **`HPOProblemWrapper` is NOT an `evox` `Problem`.** It cannot be the outer problem of a `StdWorkflow`, and no `evoxc` algorithm can be the outer optimizer.
- The outer loop is **plain host Python** — `random_search` is the only driver shipped, but any external optimizer (Optuna, scikit-optimize, a hand-written loop) works if it calls `.evaluate()`.

## API reference

`HPOProblemWrapper` and its companions are re-exported from `evox.problems` (module: `evox.problems.hpo_wrapper`).

| Name | Signature |
|---|---|
| `HPSlot` | `HPSlot(config_kind, field, transform=None)` |
| `HPOProblemConfig` | `HPOProblemConfig(algorithm, problem, hyperparameters=(), num_generations=1, monitor=HPOFitnessMonitorConfig(), num_repeats=1, seed=42, opt_direction="min", backend="numpy", device=None)` |
| `HPOFitnessMonitorConfig` | `HPOFitnessMonitorConfig(inner_monitor=EvalMonitorConfig(), fit_aggregation="mean", multi_obj_metric=None)` |
| `HPOMonitor` (base) | `HPOMonitor(num_repeats=1, fit_aggregation="mean")` → `tell_fitness(workflow)`, `aggregate(fitness)` |
| `HPOFitnessMonitor` | `HPOFitnessMonitor(num_repeats=1, fit_aggregation="mean", multi_obj_metric=None)` |
| `HPOProblemWrapper` | `HPOProblemWrapper(config)` |
| `random_search` | `random_search(wrapper, bounds, n_trials=20, seed=42) -> (best_hyperparams, best_fitness)` |

**`HPSlot`** declares one tunable field of an inner config. `config_kind` is one of `"algorithm"`, `"problem"`, `"monitor"` (the monitor slots patch the inner `EvalMonitorConfig`), `field` is the dataclass field name, and the `name` property is the dotted key `"<config_kind>.<field>"` used in mappings. `transform` is an optional **host callable** applied to the raw candidate value before substitution — e.g. `lambda v: int(round(v))` for an integer field such as `pop_size`. Candidate-vector positions follow the order of the `hyperparameters` tuple.

**`HPOProblemConfig`** is a frozen dataclass: the inner `algorithm` and `problem` config *instances*, the `HPSlot` tuple, `num_generations` per inner run, the HPO `monitor` config, `num_repeats` independent inner runs per candidate (repeat `r` uses seed `seed + r`), the base `seed`, the inner `opt_direction` (a list for multi-objective), and the inner `backend`/`device`. Validation is eager and raises `ValueError`/`TypeError`: non-dataclass inners, an unknown `config_kind`, a `field` that is not a field of that inner config, duplicate slots, `num_generations`/`num_repeats` below 1, or a non-callable `transform`/`fit_aggregation`.

**`HPOFitnessMonitorConfig` / `HPOFitnessMonitor`** pair a reused `EvalMonitorConfig` (the inner workflow's best-so-far elite tracking) with the HPO policy. `fit_aggregation` combines the per-repeat best-fitness values and is `"mean"` (default), `"min"`, `"max"`, or a callable `numpy.ndarray -> float`. `HPOMonitor.tell_fitness(workflow)` takes the inner workflow to read from (unlike torch's state-pointer version) and `aggregate(fitness)` combines a sequence of floats; subclass `HPOMonitor` to implement a custom fitness read-out.

**`HPOProblemWrapper`** exposes `num_hyperparameters` (length of the candidate vector), `get_params_keys()` (dotted names in slot order), `get_init_params()` (`{dotted name: current value}` of the inner configs), `evaluate(hyperparams=None)` and `evaluate_many(candidates)`.

- `evaluate` accepts a mapping keyed by the dotted names (missing keys keep the inner config's current value) or a vector of raw values in slot order; `None` or `{}` re-evaluates the **unchanged** inner configs. It returns one `float` (min semantics).
- `evaluate_many` takes an `(n, num_hyperparameters)` array and returns an `(n,)` array — it is a plain Python loop, **not** a batched evaluation.
- Unknown keys, a wrong vector length and non-finite values all raise `ValueError`.

**`random_search`** samples `n_trials` uniform candidates from `bounds` (one `(low, high)` pair broadcast to every slot, or one pair per slot), evaluates them through `evaluate_many`, and returns `(best_hyperparams, best_fitness)` — the best **raw** candidate vector and its fitness under minimization. Malformed bounds (`low >= high`, or neither one pair nor one pair per slot) and `n_trials < 1` raise `ValueError`.

## How a search composes with `StdWorkflow`

Each `evaluate` call is a self-contained inner run:

1. The candidate is mapped to `{slot name: value}`; each `HPSlot.transform` is applied to its raw value.
2. The values are substituted into the inner configs with `dataclasses.replace` (`algorithm.*` → the algorithm config, `problem.*` → the problem config, `monitor.*` → the inner `EvalMonitorConfig`, which is copied with **fresh, empty** history containers so host-side history never accumulates between candidates).
3. A **fresh** inner `StdWorkflow` is built with the same backend, device and `opt_direction`, plus `num_generations` from the HPO config.
4. `wf.run(num_generations, seed=seed + repeat)` is called once per repeat; after each run the wrapper reads `wf.monitor.get_best_fitness()` — or, for a multi-objective inner problem, the Pareto-front fitness `get_pf_fitness()` through `multi_obj_metric`.
5. The per-repeat values are combined with `fit_aggregation` and returned as one float.

Because the inner configs are frozen dataclasses and the wrapper holds no state between calls, `evaluate` is deterministic for a fixed candidate and seed: `wrapper.evaluate(candidate)` always reproduces the number `random_search` reports for that candidate.

## Example: tuning PSO weights on Sphere

`w` and `phi_p` are float fields of the `PSO` config (defaults `0.6` and `2.5`).

```python
from evox import EvalMonitorConfig
from evox.algorithms import PSO
from evox.problems import (
    HPOFitnessMonitorConfig,
    HPOProblemConfig,
    HPOProblemWrapper,
    HPSlot,
    Sphere,
    random_search,
)

wrapper = HPOProblemWrapper(
    HPOProblemConfig(
        algorithm=PSO(pop_size=20, lb=(-5.12,) * 3, ub=(5.12,) * 3),
        problem=Sphere(),
        hyperparameters=(HPSlot("algorithm", "w"), HPSlot("algorithm", "phi_p")),
        num_generations=20,
        monitor=HPOFitnessMonitorConfig(
            inner_monitor=EvalMonitorConfig(), fit_aggregation="mean"
        ),
        num_repeats=2,
        seed=0,
    )
)

print(wrapper.num_hyperparameters)          # 2
print(wrapper.get_params_keys())           # ['algorithm.w', 'algorithm.phi_p']
print(wrapper.get_init_params())           # {'algorithm.w': 0.6, 'algorithm.phi_p': 2.5}

# One candidate: a mapping (missing keys keep the inner default) or a vector.
print(wrapper.evaluate({"algorithm.w": 0.6}))  # 0.0012440660793799907

best, fitness = random_search(
    wrapper, bounds=[(0.1, 0.9), (1.5, 3.0)], n_trials=5, seed=0
)
print(best, fitness)  # [0.60956935 1.90468007] 0.0004765005433000624
```

`evaluate({"algorithm.w": 0.6})` leaves `phi_p` at `2.5`, so it returns the same value as `evaluate()` (the unchanged default config); `evaluate([0.6, 2.5])` is equivalent to both. The search only sampled five candidates, so treat the "best" vector as illustrative — with `num_repeats=2` each candidate costs two inner runs of twenty generations.

## Example: a multi-objective inner problem

For a multi-objective inner workflow the single-objective accessor is meaningless, so `multi_obj_metric` condenses the Pareto front to a number, and the inner monitor needs `full_fit_history=True` so `get_pf_fitness()` has a front to hand over.

```python
import numpy as np

from evox import EvalMonitorConfig
from evox.algorithms import NSGA2
from evox.problems import (
    DTLZ2,
    HPOFitnessMonitorConfig,
    HPOProblemConfig,
    HPOProblemWrapper,
    HPSlot,
)

mo_wrapper = HPOProblemWrapper(
    HPOProblemConfig(
        algorithm=NSGA2(pop_size=20, n_objs=2, lb=(0.0,) * 4, ub=(1.0,) * 4),
        problem=DTLZ2(d=4, m=2),
        hyperparameters=(
            HPSlot("algorithm", "pop_size", transform=lambda v: int(round(v))),
        ),
        num_generations=10,
        monitor=HPOFitnessMonitorConfig(
            inner_monitor=EvalMonitorConfig(full_fit_history=True),
            multi_obj_metric=lambda pf: np.linalg.norm(np.asarray(pf), axis=1),
        ),
        opt_direction=["min", "min"],
        seed=0,
    )
)
print(mo_wrapper.get_init_params())  # {'algorithm.pop_size': 20}
print(mo_wrapper.evaluate([20.0]))   # 1.0008732308883508
```

The DTLZ2 Pareto front is the unit sphere, so the smallest achieved norm is ≈ 1 and the metric's minimum is a sensible quality score. The `numpy` backend prints a benign `RuntimeWarning: divide by zero` from the crowding-distance normalisation (a zero-range objective column) — it does not affect the selection and the test suite filters exactly this warning.

## Limitations

Only what the module documents, but all of it is load-bearing:

- **No gradient-based or in-graph HPO.** There is no outer `StdWorkflow`: candidates are evaluated one at a time by host Python.
- **Per-trial re-trace/re-compile cost dominates.** Config values are static leaves baked into a trace, so every *distinct* candidate re-traces and re-compiles the inner step graph; throughput is dominated by build time rather than by the optimization itself. `num_repeats` reuses **one** built workflow per candidate (only the seed changes).
- **No batched instances.** `num_instances` + `vmap` parallelism has no analogue here; `evaluate_many` is a sequential loop.
- **Only plain static config fields are tunable.** Hyperparameters must be Python scalars, flat float tuples or strings that `dataclasses.replace` can substitute — tensor-valued or otherwise traced hyperparameters are unsupported, and only fields of the inner algorithm/problem/monitor configs can be tuned (there is no matrix-valued hyperparameter support).
- **The inner monitor must be an `EvalMonitorConfig`.** `HPOFitnessMonitor.tell_fitness` raises if the inner workflow was built without one.
- **Multi-objective inner problems need `multi_obj_metric`** (the single-objective accessor is meaningless there) **plus `full_fit_history=True`** on the inner monitor, and `opt_direction` must be a list. The metric is a **host-side callable over a numpy `(n_pareto, n_objs)` array** returning something `np.min`-able; an `evoxc` metric such as `evox.metrics.igd` cannot run outside a trace and must first be wrapped host-side with `evoxc.build` + `evoxc.run` (see {doc}`metrics`).
- **No torch plumbing.** `HPOData`, the custom-op evaluate loop and `copy_init_state` have no analogue: the inner workflow is rebuilt per candidate, so there is no shared init state to copy.

```{tip}
Budget for recompilation, not for generations. Each candidate pays a full inner trace/build, so prefer **few generations per trial** (10–20) and a **modest `n_trials`** (5–20) while you tune; scale up only the winning region afterwards. Use the `numpy` backend for the inner workflow while searching — the search itself is serial and host-bound, so a GPU backend adds transfer/compile overhead without any parallel speedup. For a fast hand-off, evaluate the inner workflow with a cheap surrogate problem (e.g. `Sphere`) first and only re-run the shortlist on the expensive problem.
```

```{note}
Looking for the tunable fields of an algorithm, problem or monitor? They are exactly the fields of its frozen config dataclass — read the config's API page, or `dataclasses.fields(...)` at runtime; a wrong slot name fails fast when the `HPOProblemWrapper` is constructed. For custom algorithms/problems see {doc}`../developer/custom-alg-pro`, and for a full workflow walk-through see {doc}`../../tutorial/tutorial_part3`.
```
