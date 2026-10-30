# 3. Workflow and Monitor

{doc}`Part 2 <tutorial_part2>` showed the shortest path from a config to a result. This
chapter explains the machinery behind it: what `StdWorkflow` does with each of its
arguments, how a generation maps onto an algorithm step, how state and randomness are
threaded, and how `EvalMonitorConfig`/`EvalMonitor` record and expose the run.

## The `StdWorkflow` contract

`StdWorkflow` is the only orchestration object in EvoX 2.0. It takes frozen config
dataclasses, traces them into a compiled step graph, and runs that graph generation by
generation:

| Argument | Type | Role |
|---|---|---|
| `algorithm=` | frozen config dataclass | Algorithm to run. Its module must define `step(config, state, evaluate)`, and optionally `init_step`/`final_step`. |
| `problem=` | frozen config dataclass | Problem to evaluate. Its module defines `evaluate(config, state, x) -> (fitness, new_state)`. |
| `monitor=` | config dataclass or `None` (default) | Monitor to update every generation. Without one, `wf.monitor` stays `None`. |
| `opt_direction=` | `str` or `list[str]` (default `"min"`) | `"min"`/`"max"`, or a list of them for multi-objective problems. |
| `solution_transform=` | callable or `None` (default) | Applied to the candidates **before** they reach the problem. |
| `fitness_transform=` | callable or `None` (default) | Applied to the fitness **after** opt-direction scaling, before the algorithm sees it. |
| `num_generations=` | `int` or `None` (default) | Default budget for `run()`/`fit()`. |
| `backend=` | `str` (default `"numpy"`) | `evoxc` compiler backend for the step graph. |
| `device=` | `str`/`evoxc.Device`/`None` (default) | Device the step graph runs on; `None` means CPU. |
| `compile_options=` | `dict` or `None` (default) | Extra keyword options for `evoxc.build` (`backend`/`device` keys are stripped). |

Both transforms must be **plain, traceable callables** — they run *inside* the compiled
graph, so a stateful object (anything that keeps mutable Python attributes) or a function
that reads a Python global at trace time will not work. The evaluation pipeline they slot
into is fixed, and it is the only place fitness is produced:

```text
evaluate(candidates):
    1. x          = solution_transform(candidates)        # if given
    2. fitness, problem_state = problem.evaluate(problem_state, x)
    3. fitness    = opt_direction * fitness               # min semantics inside
    4. fitness    = fitness_transform(fitness)            # if given
    5. monitor_state = monitor_update(candidates, fitness)  # raw candidates!
    return fitness
```

Two consequences are worth remembering:

- The monitor records the **raw** candidates (step 1 input) together with the
  **transformed** fitness (step 4 output). If `solution_transform` changes the meaning
  of a point, the reported solution is the search-space point you asked for, not the
  point the problem actually saw.
- Everything downstream — the algorithm's own bookkeeping as well as the monitor — sees
  the transformed fitness, and always in minimization orientation (step 3).

```python
import numpy as np

from evox_etl import EvalMonitorConfig, StdWorkflow
from evox_etl.algorithms import PSO
from evox_etl.problems.numerical import Sphere

dim = 3
lb = np.zeros(dim)
ub = np.full(dim, 10.0)

# Sphere is minimized at the origin, but we want the optimum at x = 3
wf = StdWorkflow(
    PSO(pop_size=40, lb=lb, ub=ub),
    Sphere(),
    monitor=EvalMonitorConfig(),
    solution_transform=lambda x: x - 3.0,
    backend="numpy",
)
wf.run(100, seed=0)
print(f"best fitness : {wf.monitor.get_best_fitness():.5f}")
print(f"best solution: {np.round(wf.monitor.get_best_solution(), 4)}")
```

```text
best fitness : 0.00092
best solution: [3.0135 3.0221 3.0158]
```

## `run` vs `fit` vs the step loop

| Call | What it does | Returns | Needs a monitor? |
|---|---|---|---|
| `wf.run(generations=None, seed=42)` | `init` + the whole loop (`init_step`, plain `step`s, `final_step`) | the final `WorkflowState` | no |
| `wf.fit(fitness=None, generations=None, seed=42)` | the same loop, then reads the elite | the best fitness (`float`) | yes — raises `ValueError` without one |
| `wf.init(seed=42)` | builds the init graph (and the plain `step` graph) and returns the generation-0 state — no generation is evaluated | the initial `WorkflowState` | no |
| `wf.init_step(state)` / `wf.step(state)` / `wf.final_step(state)` | one generation per call; you own the loop | a new `WorkflowState` per call | no |

The generation count is resolved per call: `run(n)`/`fit(generations=n)` override the
constructor's `num_generations`. When neither is given you get a clear error rather than
a silent default:

```text
ValueError: The number of generations must be specified either in the constructor
(num_generations) or in run(generations=...).
```

`fit()` is the shortest path to a number, and its optional `fitness=` argument records
an externally known fitness vector as the generation-0 history entry (useful when you
resume a run from a warm start):

```python
best = wf.fit(fitness=np.full(20, 1.0), generations=10, seed=0)   # -> Python float
```

The manual loop is the escape hatch — use it for early stopping, per-generation
logging, checkpointing, or for interleaving a second experiment:

```python
state = wf.init(seed=42)
state = wf.init_step(state)
for _ in range(2, 201):
    if wf.monitor.get_best_fitness() < 1e-8:      # early stop
        break
    state = wf.step(state)
wf.flush()
```

## One generation = one algorithm step

An algorithm module exposes `init(config, key)` and `step(config, state, evaluate)`.
A single `step` call owns **one full generation**: it generates the candidates, asks
`evaluate` for their fitness, updates its state, and returns the new state. It may call
`evaluate` more than once per generation (CoDE evaluates three trial batches; that is an
algorithm detail, not a workflow one).

`evaluate` is an opaque closure created by the workflow — treat it exactly like a
function from a candidate batch to fitness. Do not store it in your state, do not
re-thread it through return values, and do not try to read the problem or monitor state
from it: the workflow owns that state, keeps the state of the last `evaluate` call, and
increments the generation counter itself.

The workflow also owns the **compile site**: the step graph is built once per step
variant (`init_step`/`step`/`final_step`) and per state-shape signature, then reused for
every generation. A step that changes the shape of its state leaves (a shrinking
population, a growing elite buffer) simply triggers one re-trace — you never manage
compilation yourself.

## `opt_direction` semantics

All algorithms minimize internally. `opt_direction="max"` does not change the algorithm;
the workflow multiplies the fitness by `-1` before it reaches the algorithm and undoes
that scaling in every monitor accessor, so you always read direction-corrected numbers:

```python
wf = StdWorkflow(
    PSO(pop_size=40, lb=np.full(3, 5.0), ub=np.full(3, 10.0)),
    Sphere(),
    monitor=EvalMonitorConfig(),
    opt_direction="max",
    backend="numpy",
)
wf.run(100, seed=0)
print(wf.opt_direction)               # (-1,)  — the baked-in scaling
print(f"{wf.monitor.get_best_fitness():.1f}")   # 300.0, the MAXIMUM of Sphere
```

For multi-objective problems pass a list with one entry per objective, e.g.
`opt_direction=["min", "max"]`. That list is also what tells the workflow (and the
monitor) that a run is multi-objective: `multi_obj` and `n_obj` are derived from its
length.

## State threading

Every state is an immutable value: `step` never mutates in place, it returns a new
state. Keep the return value:

```python
state = wf.init(seed=42)
state = wf.init_step(state)
state = wf.step(state)      # explicit threading: clearest
state = wf.step()           # also fine: the workflow remembers the latest state
wf.flush()
print(int(state.generation.numpy()))            # 3 — one increment per step
print(len(wf.monitor.fitness_history))          # 3
```

Passing `state=None` (or nothing) uses the state the workflow last returned. Explicit
threading is still preferred: it survives interleaving two workflows, and it makes it
obvious where a generation boundary is. `state.generation` is a 0-d `int32` tensor and
`state.key` the RNG key; `state.algorithm_state`, `state.problem_state` and
`state.monitor_state` are the component states.

## Stateless RNG

There is **no global seed and no hidden RNG state**: randomness comes from explicit
keys. `evox_etl.key(seed)` builds a key, `evox_etl.random` (the `evoxc.random` module)
draws from it, and keys travel inside the states. The random ops are graph ops, so you
sample from a key inside traced code — `evoxc.evaluate` runs a graph definition eagerly
and returns a concrete tensor:

```python
import evoxc

from evox_etl import key, random

def sample(k):
    k, subkey = random.split(k)      # split, use the subkey, keep the parent
    return random.uniform(subkey, (3,), low=-5.0, high=5.0)

print(evoxc.evaluate(sample, key(42)).numpy())   # [ 2.8856735 -2.4464002 -0.3658264]
print(evoxc.evaluate(sample, key(42)).numpy())   # identical — same key, same numbers
```

`random` provides `key`, `split`, `split_n`, `uniform`, `normal`, `randint`,
`permutation` and `multinomial`. Algorithms follow one convention: each `step` splits a
subkey off the key stored in its own state for the generation's draws and stores the
parent key back, so a given state reproduces bit-identical numbers forever. That is what makes `seed=` meaningful —
`run(50, seed=42)` is exactly `wf.init(seed=42)` followed by the loop, and the same seed
gives the same run.

## `EvalMonitorConfig`

`EvalMonitorConfig` is a frozen dataclass, but its history fields are host-side
containers that the workflow appends to after each generation:

| Field | Default | Who fills it | Meaning |
|---|---|---|---|
| `multi_obj` | `None` | workflow | Multi-objective switch, derived from a **list** `opt_direction`. |
| `full_fit_history` | `True` | you | Record every generation's fitness vector. |
| `full_sol_history` | `False` | you | Record every generation's candidates. |
| `full_pop_history` | `False` | you | Record every generation's population; also gates the auxiliary channel. |
| `topk` | `1` | you | How many elite solutions the single-objective monitor tracks. |
| `pop_size` | `None` | workflow | Buffer size, inferred from the algorithm state (`population`/`pop`) or config. |
| `dim` | `None` | workflow | Solution dimension; inferred the same way — pass it yourself for CMA-ES. |
| `n_obj` | `None` | workflow | Number of objectives (multi-objective only). |
| `opt_direction` | `(1,)` | workflow | Per-objective `±1` scaling; the workflow's direction always wins. |
| `fit_history` | `[]` | workflow | Per-generation fitness arrays. |
| `sol_history` | `[]` | workflow | Per-generation solution arrays. |
| `pop_history` | `[]` | workflow | Per-generation populations (legacy fallback; only used when the algorithm module has no `record_step` hook). |
| `aux_history` | `{}` | algorithm hook | Named auxiliary tensors, each a list of per-generation arrays. |

The `full_*_history` flags are not free: each enabled flag keeps one array per
generation on the host for the lifetime of the config, so memory grows as
`generations × pop_size × dim`. Leave `full_sol_history`/`full_pop_history` off unless
you actually need the populations (their main consumer is the multi-objective Pareto
front in {doc}`Part 6 <tutorial_part6>`). `aux_history` only receives data when
`full_pop_history=True` **and** the algorithm module provides a `record_step` hook.

## `EvalMonitor` methods

`wf.monitor` is created by the workflow during `init()` and refreshed after every step;
its `.config` is the completed config and its `.state` the current monitor state.

| Method / property | Returns | Notes |
|---|---|---|
| `get_best_fitness()` | `float` | Best so far, direction-corrected. Single-objective only. |
| `get_best_solution()` | `(dim,)` array | The elite that achieved `get_best_fitness()`. Single-objective only. |
| `get_topk_fitness()` | `(topk,)` array | The top-`topk` elite fitness values. Single-objective only. |
| `get_topk_solutions()` | `(topk, dim)` array | The matching elite solutions. Single-objective only. |
| `get_latest_fitness()` | last generation's fitness batch | Not necessarily `topk`-related. |
| `get_latest_solution()` | last generation's candidate batch | Raw candidates (pre-`solution_transform`). |
| `get_fitness_history()` | `list` of arrays | One entry per generation, direction-corrected. |
| `get_pf()` | `(solutions, fitness)` | Pareto front. Multi-objective only; needs `full_fit_history` + `full_sol_history`. |
| `get_pf_fitness()` | `(k, n_obj)` array | Pareto-front fitness. Multi-objective only; needs `full_fit_history`. |
| `record_auxiliary(aux)` | `None` | Record one generation of algorithm auxiliary tensors (gated on `full_pop_history`). |
| `aux_history` | `dict[str, list]` | The recorded auxiliary channel (`auxiliary_history` is an alias). |
| `get_pf_solutions()` | `(k, dim)` array | Pareto-front solutions. Multi-objective only; needs `full_fit_history` + `full_sol_history`. |
| `plot(problem_pf=None, source="eval", **kwargs)` | Plotly `Figure` | 1-D convergence curve for single-objective, 2-D/3-D objective-space scatter for multi-objective. |

The accessors are split by problem type, and misuse is an error rather than a silent
wrong answer: `get_best_*`/`get_topk_*` raise
`ValueError: Multi-objective optimization does not have a single best solution...` under
multi-objective, and `get_pf*` raise under single-objective. Every accessor returns
host-side NumPy data (and `get_best_fitness()` a Python `float`), so `print`, `np.round`
and `float(...)` all work directly — no `.item()` call is needed.

## Where monitors run

The monitor is a module like any other component: `monitor_update(config, state,
candidate, fitness)` is a plain function that is **traced into the step graph**, which
is why the update itself costs nothing per generation beyond the tensors it touches.
Everything you actually read from is **host-side**: the history lists, the Pareto front
computation and the plots are computed from the state leaves that are copied back after
each step, and the device→host copies are pipelined one generation behind the step graph
so they overlap with computation.

That pipelining is visible in exactly one place: after driving the loop yourself on a
non-CPU `device=`, the last generation's history entry is still in flight until you call

```python
wf.flush()    # block until every deferred monitor readback has been appended
```

`run()` and `fit()` flush at the end of their own loop, so they never need it. On the
default CPU setup (`device=None`) each entry is appended to the history immediately, so
`flush()` is a no-op there and the lists are already complete without it.

Because the per-generation update is compiled, there is no hook for arbitrary host-side
side effects (no per-generation callback that opens a file, prints, or mutates a Python
object). The one supported channel is declarative: an algorithm module may define
`record_step(config, state, candidate, fitness)` returning a dict of concrete tensors,
and the workflow hands that dict to `monitor.record_auxiliary(...)` every generation,
where it lands in `aux_history` under the returned keys.

## Next steps

The protocols behind all of this — the step protocol, the problem and monitor
functions, and what a component module must define — are specified in
{doc}`the architecture guide <../guide/developer/architecture>`. To write your own
monitor (a custom metric, a convergence log, an early-stopping condition), continue
with {doc}`Writing a custom monitor <../guide/developer/custom-monitor>`.
