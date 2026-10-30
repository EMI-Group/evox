# Architecture of EvoX 2.0

EvoX 2.0 is **data plus functions**. A component — algorithm, problem or monitor — is a
frozen config dataclass that says *what it is*, plus plain module-level functions that say
*what it does*. That explains most of the design: no base class to inherit from, no
per-component compile or vectorisation call, no state hidden in object attributes. A run
is a value (the config) threaded through functions that return new values (the states).
`evox_etl` ships 30+ algorithms and 30+ benchmark problems that follow the conventions
below.

```{note}
This page documents the **experimental EvoX 2.0 pre-release**, import name `evox_etl`,
built on the [EvoX Compiler](https://github.com/EMI-Group/evoxc) (`evoxc`). The sibling
pages show how to write each piece.
```

## Three protocols, one resolution rule

`evox_etl.core` defines three `typing.Protocol` classes — `Algorithm`, `Problem` and
`Monitor` — plus the aliases their signatures use (`AlgorithmConfig`,
`AlgorithmState`, `Candidates`, `Fitness`, `Evaluate`, `KeyArray` and the problem/monitor
equivalents). They are **documentation only**: no component inherits from them and
nothing checks them at runtime. `WorkflowState` and `EmptyState` are the two concrete
state containers exported alongside them.

The binding rule is how `StdWorkflow` finds a component's functions: it receives a config
*instance*, resolves the module its class was defined in, and looks up well-known names
there.

```python
import importlib
module = importlib.import_module(type(config).__module__)   # the config's own module
step = module.step                                          # well-known name
```

Two consequences follow:

- The config and its functions **must live in the same module** — that is the only link
  between them.
- **Each component needs its own module.** `init`, `step` and `evaluate` are plain
  module-level names, so two components sharing a file would shadow each other.

```{warning}
`StdWorkflow.__init__` raises `TypeError` for anything that is not a dataclass: pass a
config **instance**, not a class, a module or a callable. A missing `step` is reported the
same way, at construction rather than inside a trace.
```

## Config vs state

| | **Config** | **State** |
|---|---|---|
| Python type | frozen dataclass (`@dataclasses.dataclass(frozen=True)`) | namedtuple or frozen dataclass |
| Leaves | static only: Python scalars, strings, `None`, flat tuples of floats | `evoxc` tensors only |
| NumPy arrays | never stored — normalised host-side in `make_*` | not applicable |
| Shapes | not applicable | fixed; a resizing tensor leaf forces a fresh trace |
| Handed to functions | first argument, by value (static) | second argument (a pytree) |
| Changed after construction | never | never — a function returns a new state |

A config is *fully static*: it carries no tensors, so the compiler specialises on it at
trace time. Array-like hyperparameters are therefore
normalised to flat float tuples, and values such as bounds are materialised inside the
traced function instead:

```python
lb = evoxc.ops.constant(evoxc.core.tensor(np.asarray(config.lb, dtype=np.float32)))
```

Callables are allowed on a config only when it registers its own pytree node (the
multi-objective algorithm configs accepting custom operators do so with
`evoxc.register_pytree_node`); elsewhere a non-`None` callable field is rejected.

A state is the opposite: every leaf is a concrete `evoxc` tensor with a fixed shape and
dtype, and it is the only thing that varies between generations. Being an ordinary pytree,
a state can be mapped over, copied, moved to another device and serialised — the workflow
does all four — and algorithm states carry an RNG key leaf (see [Randomness](#randomness)). The
workflow's own `WorkflowState` has five fields: `algorithm_state`, `problem_state`,
`monitor_state`, `generation` (a 0-d `int32` tensor) and `key` (0-d `int64`).

## The step protocol

Algorithms, problems and monitors each implement a small set of plain functions living
next to their config.

| Algorithm function | Signature | Required | Called by |
|---|---|---|---|
| `init` | `init(config, key) -> state` | yes | the init graph, traced once |
| `init_step` | `init_step(config, state, evaluate) -> state` | no | generation 1 (falls back to `step`) |
| `step` | `step(config, state, evaluate) -> state` | **yes** | every other generation |
| `final_step` | `final_step(config, state, evaluate) -> state` | no | the last generation (falls back to `step`) |
| `record_step` | `record_step(config, state, candidate, fitness) -> dict[str, tensor] \| None` | no | host-side, after each generation |

| Problem function | Signature | Required | Called by |
|---|---|---|---|
| `evaluate` | `evaluate(config, state, pop) -> (fitness, state)` | **yes** | inside the evaluate closure |
| `init` | `init(config, key) -> state` | no | the init graph (stateful problems only) |

| Monitor function | Signature | Required | Called by |
|---|---|---|---|
| `monitor_update` | `monitor_update(config, state, candidate, fitness) -> state` | **yes** | inside the evaluate closure |
| `init` | `init(config, key) -> state` | no | the monitor init graph |
| `record_auxiliary` | `record_auxiliary(config, aux)` | no | host-side, after each generation |
| `Monitor` | wrapper class, `Monitor(config=..., state=...)` | no | host-side, created during `init` |

`step` owns **one whole generation**: generate candidates → `evaluate(...)` → update
state → return it. The workflow does not know what an algorithm does; it only decides
which variant runs — generation 1 is `init_step`, the last is `final_step` (each falling
back to `step`), everything between is a plain `step`, so `run(generations=1)` performs
only the `init_step`. `record_step` is the one algorithm hook that runs on the **host**:
it gets the post-step state plus the monitor's `latest_solution`/`latest_fitness` (or
`None`) and returns a dict of **existing** tensors for the monitor's `record_auxiliary` —
creating a new tensor there raises `evoxc`'s `TraceError`.

```{warning}
These must be **plain functions**, not `@evoxc.defn` objects. Defn objects raise when
called, even inside a trace — the workflow calls every graph function directly.
```

## The evaluate closure

An algorithm never sees the problem. It receives `evaluate`, a traced closure injected by
the workflow, and calls it with a candidate batch. The pipeline is fixed:

```text
evaluate(candidates):
    1. x            = solution_transform(candidates)              # if configured
    2. fitness, ps  = problem.evaluate(problem_config, ps, x)
    3. fitness      = opt_direction_constant * fitness            # always minimised
    4. fitness      = fitness_transform(fitness)                  # if configured
    5. monitor_state = monitor_update(monitor_config, monitor_state,
                                      raw_candidates, fitness)
    return fitness
```

Three consequences matter when writing a component:

- **Several calls per generation are fine.** A `step` may call `evaluate` with different
  candidate sets (trial batches, restarts, sub-populations); the workflow keeps the
  problem and monitor state of the **last** call.
- **The monitor sees raw candidates but transformed fitness.** Step 1 happens first, so a
  recorded solution is the search-space point, not the one the problem evaluated; its
  fitness is direction-scaled and passed through `fitness_transform`.
- **The closure is opaque and transient.** Never store it in a state, re-thread it through
  return values, or read problem/monitor state from it: the workflow owns both.

## What `StdWorkflow` builds

1. **Construction.** Validate the three configs (dataclasses, and a callable `step` in
   the algorithm's module) and store the `backend`, `device` and `compile_options`.
2. **`init(seed=...)`.** Compile the init graph once with `evoxc.build` on the numpy
   backend and CPU, and run it: the algorithm's `init` plus the problem's `init` when the
   problem is stateful. The host side then completes the monitor config and runs a tiny
   monitor init graph.
3. **Lazy step compilation.** One step graph per *(resolved variant, state leaf signature)*
   is compiled and cached. The signature is the ordered tuple of `(shape, dtype)` per
   tensor leaf, so a state that legitimately resizes — a shrinking population, a growing
   elite buffer — triggers one fresh trace and then runs at full speed again.
4. **The host loop.** `run`/`fit` dispatch `init_step` → `step` × (generations − 2) →
   `final_step`, one compiled graph run per generation.
5. **History.** The monitor's history lists live on the config and are filled host-side.
   Device→host copies are pipelined one generation behind, overlapping the next
   generation's graph, and drained at the end of `run`/`fit` (or by `flush()`).

Behind steps 2 and 3 is the compiler's own pipeline: `evoxc.build` =
**trace → lower → compile → load**. The workflow is the only place that builds anything,
which is why components contain no compilation logic. Before the monitor init graph runs,
the workflow fills in the fields it can infer with `dataclasses.replace`:

| Monitor field | Filled with |
|---|---|
| `pop_size` | the algorithm state's `population` (or `pop`) tensor's first dimension, else the algorithm config's `pop_size` |
| `dim` | that tensor's second dimension, else the algorithm config's `dim` |
| `n_obj` | `len(opt_direction)` when `opt_direction` is a list |
| `multi_obj` | `len(opt_direction) > 1` when it is left auto (`None`) |
| `opt_direction` | always the workflow's authoritative tuple of `1`/`-1` ints |

```{tip}
The workflow's direction always wins, so a monitor that reports fitness back to users should
un-negate it for `"max"` — the built-in `EvalMonitor` does exactly that.
```

## Randomness

Keys are ordinary values, not a global generator. `StdWorkflow.init` starts from
`evoxc.random.key(seed)`; the workflow keeps its own key and hands a split to the
algorithm's `init`. Each algorithm state then stores its own key, splits it once per
generation, and stores the parent back:

```python
key, subkey = evoxc.random.split(state.key)   # draw from subkey, store key back
```

There is no global RNG and no in-place mutation, so the same seed with the same config
reproduces the same run bit for bit, and any intermediate state can be resumed exactly.

## The `make_*` convention

Some configs validate their own fields in `__post_init__`, but array normalisation and
derived defaults are host-side work, so the friendly entry point is a module-level
`make_<component>` constructor in the same module. It runs on
the **host, outside any trace**, and does three things: normalise array-like input
(bounds, centres, weights) into flat float tuples; validate (positivity, shape agreement,
ordering) with a clear `ValueError`; and derive defaults that must appear as fields, such
as a population size computed from the dimension. It returns the frozen config. Shared
helpers live in `evox_etl.algorithms._config_utils` (`to_float_tuple`,
`normalize_bounds`, `require_gt`, `bake_bounds`, …); direct construction works for
already-normalised values but skips the `make_*` normalisation and validation.

## Why this design

- **Nothing to keep in sync.** No marking layer declares which values are fixed and which
  vary: a config leaf is static, a state leaf is a tensor, and the compiler enforces it.
- **State is plain data.** A run is a pytree of tensors: map over it, copy it, move it
  between devices, serialise it — no framework support needed.
- **Exactly one compile site.** Components never trace themselves, so every tensor
  operation ends up inside a graph — there is no eager path to fall back to.
- **Backend and device are a workflow-level choice.** The same configs run unchanged
  elsewhere: see {doc}`../advanced/backends` for the backends and
  {doc}`../../tutorial/tutorial_part3` for the workflow and monitor mechanics.

## Verified config/state excerpt

Static hyperparameters on one side, tensor leaves on the other:

```python
import dataclasses
import evoxc

Tensor = evoxc.SymbolicTensor


@dataclasses.dataclass(frozen=True)
class GaussianES:
    """Hyperparameters: plain Python values, fixed for the whole run."""

    pop_size: int
    dim: int
    sigma: float = 0.1
    lb: float = -10.0
    ub: float = 10.0


@dataclasses.dataclass(frozen=True)
class GaussianESState:
    """Algorithm state: evoxc tensor leaves only."""

    population: Tensor  # (pop_size, dim)
    fitness: Tensor  # (pop_size,)
    best_solution: Tensor  # (dim,)
    best_fitness: Tensor  # ()
    key: Tensor  # ()
```

## Next

- {doc}`custom-alg-pro` — write your own algorithm and problem.
- {doc}`custom-monitor` — write your own monitor and read a run back on the host.
- {doc}`../advanced/backends` — run the same configs on another backend or device.
- {doc}`../../tutorial/tutorial_part3` — the workflow and monitor, step by step.
