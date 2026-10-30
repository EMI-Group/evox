# Writing a custom monitor

A monitor observes every generation of an optimization run. It sees the **raw
candidate batch** the algorithm hands to `evaluate` together with the **transformed
fitness** (already scaled into minimization orientation), keeps tensor state inside the
compiled step graph, and exposes results on the host.

EvoX 2.0 ships `EvalMonitorConfig`/`EvalMonitor` as the default monitor. This page
shows how to write your own: a frozen config dataclass plus plain module-level functions
in the same module.

```{note}
The built-in monitor and the `run()`/`fit()` loop are covered in
{doc}`../../tutorial/tutorial_part3`; start there if you only need to read a run back.
```

## The Monitor protocol

There is no base class to inherit from. A monitor is a **plain Python module**, and
`StdWorkflow` resolves its functions with
`importlib.import_module(type(config).__module__)`: the config object decides which
module is called, so every function the workflow needs must live in the **same module**
as the config class. The module may define:

| Definition | Required | Where it runs | Role |
|---|---|---|---|
| config dataclass (frozen) | yes | host + trace | Static leaves, plus host-side history lists (plain Python lists on the config, appended by the workflow or by your host-side hook). |
| `init(config, key) -> state` | no | once, in a traced init graph | Allocate the fixed-shape monitor state. The `key` is a split random key, available if the monitor needs randomness. |
| `monitor_update(config, state, candidate, fitness) -> state` | yes | inside every generation's step graph | Consume one candidate batch and its fitness, return the next monitor state. |
| `record_auxiliary(config, aux) -> None` | no | host-side, once per generation | Store the dict returned by the algorithm module's `record_step` hook. Called only when the algorithm module also defines `record_step`. |
| `Monitor` class | no | host-side | Convenience wrapper constructed by the workflow as `Monitor(config=..., state=...)`; `wrapper.state` is refreshed after every generation. |

A few details matter:

- The config is frozen, but its history *containers* are ordinary mutable Python
  objects: "frozen" only prevents rebinding the fields, while the host-side path
  appends to list fields.
- The graph functions are plain module-level functions (not `evoxc` definitions).
  They are called inside an active trace; calling them outside one raises the
  compiler's `TraceError`.
- The `Monitor` wrapper is optional, but the built-in one exposes
  `get_best_fitness`/`get_best_solution`, and `workflow.fit()` requires a wrapper with
  `get_best_fitness`.
- The protocol class in `evox_etl.core.monitor` (also named `Monitor`) is
  **documentation only**: monitors are duck-typed, and no inheritance is involved.

## What the workflow fills in for you

Before your `init` runs, `StdWorkflow` completes the monitor config with
`dataclasses.replace`, touching only the fields the config actually declares: `pop_size`,
`dim`, `n_obj`, `multi_obj` and `opt_direction`. {doc}`architecture` holds the
field-by-field table of what the workflow supplies.

A 2-D `population`/`pop` tensor supplies both `pop_size` and `dim`; a 1-D one supplies
only `pop_size`, with `dim` falling back to the algorithm config. The practical
consequence: if your monitor needs a buffer shape, **declare `pop_size` and `dim`** on
your config (both defaulting to `None`) and raise a clear `ValueError` in `init` when
they are still `None` — the example below does exactly that.

```{tip}
The workflow only fills field names it knows; extra fields are yours to populate and
manage.
```

## How `monitor_update` is called

`monitor_update` runs **inside the compiled step graph**, once for every `evaluate(...)`
call the algorithm makes during a generation:

- `candidate` is the raw candidate batch the algorithm handed to `evaluate`, *before*
  any `solution_transform`.
- `fitness` is the transformed fitness: opt-direction scaled and then passed through
  `fitness_transform`, so it is always in minimization orientation.

Return a **new** state object and never mutate the old one: state is threaded through
the graph as an immutable value. Keep the tensor leaves at fixed shapes — a step graph
is traced for one state signature and cached, so stable shapes let the graph be reused
for the whole run. Resizing a leaf makes the workflow trace a fresh graph, a re-trace
you pay for, so resize only when you are willing to accept it.

```{warning}
The workflow keeps the monitor state produced by the **last** `evaluate` call of a
generation. An algorithm that evaluates several candidate batches per step still updates
the monitor on each call, but only the last state survives — so a monitor records once
per generation, not once per candidate batch.
```

## Host-side access

In-graph code cannot print arrays or hand them to NumPy. Do that in the host-side
wrapper, reading a concrete tensor leaf with:

```python
tensor.to(evoxc.core.Device("cpu")).numpy()
```

History lists belong on the **config** as plain Python lists (for example
`fit_history`). For the built-in `EvalMonitorConfig`, the workflow appends
`latest_fitness`, `latest_solution`, and — only when the algorithm module has no
`record_step` hook — population entries, each gated by its `full_*_history` flag. Your
own config can use the same pattern, as the example does with `aux_history`.

The workflow exposes `workflow.monitor` (the wrapper, or `None`) and
`workflow.monitor_config` (the completed config). `workflow.monitor_state` holds the state
captured during initialization; only the wrapper's `.state` is refreshed after every
generation, so read the wrapper rather than that attribute.

`workflow.run()` returns the final `WorkflowState`. `workflow.fit()` runs the same loop
and returns `workflow.monitor.get_best_fitness()`, raising a `ValueError` if there is no
wrapper with that method.

## The auxiliary channel (`record_auxiliary` / `aux_history`)

The monitor state is restricted to fixed-shape tensor leaves, so it is a poor place for
arbitrary diagnostics. The auxiliary channel covers that case. It works like this:

1. An algorithm module **may** define a plain host-side function
   `record_step(config, state, candidate, fitness) -> dict[str, tensor] | None`. After
   every generation, outside any trace, `StdWorkflow` calls it with the post-step
   algorithm state and the monitor's `latest_solution`/`latest_fitness` (or `None` when
   those leaves are absent). Returning `None` skips the monitor call.
2. If — and only if — the algorithm defines `record_step` **and** the monitor module
   defines `record_auxiliary`, the workflow hands that dict to
   `record_auxiliary(config, aux)` once per generation.

`record_step` runs on the host with concrete, already-computed tensors. It must return
existing state leaves; a tensor operation there raises the compiler's `TraceError`,
because there is no eager mode to fall back on. A valid hook is just a projection:

```python
def record_step(config, state, candidate, fitness):
    """Optional host-side hook: one dict entry per aux_history key."""
    return {"best_fitness": state.best_fitness, "population": state.population}
```

The built-in `EvalMonitor.record_auxiliary` stores each value as a NumPy array under
`config.aux_history[name]` once per generation, but only when
`EvalMonitorConfig.full_pop_history=True`; its `pop_history` property then prefers
`aux_history["pop"]` over the legacy list. A custom monitor can store unconditionally,
as the example does.

```{note}
The auxiliary channel is per generation, host-side, and outside the compiled graph —
the right place for figures, diagnostics, or metrics that would be expensive to compute
inside the trace.
```

## A complete, verified custom monitor

The module below tracks the best-so-far solution and fitness and records auxiliary
history. Save it as `my_monitor.py`.

```python
"""A minimal best-so-far monitor written as an evox_etl monitor module."""

from __future__ import annotations

import dataclasses
from dataclasses import field
from typing import Any

import evoxc
import evoxc.core
import evoxc.numpy as enp
import numpy as np

Tensor = evoxc.SymbolicTensor


def _constant(array) -> Tensor:
    """Bake a host numpy array into an in-graph constant."""
    return evoxc.ops.constant(evoxc.core.tensor(np.asarray(array, dtype=np.float32)))


@dataclasses.dataclass(frozen=True)
class BestMonitorConfig:
    """Static config; ``pop_size``/``dim``/``opt_direction`` are filled by the workflow."""

    pop_size: int | None = None
    dim: int | None = None
    opt_direction: tuple = (1,)
    aux_history: dict = field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class BestMonitorState:
    """Monitor state: fixed-shape tensor leaves only."""

    best_solution: Tensor  # (dim,)
    best_fitness: Tensor  # ()
    latest_solution: Tensor  # (pop_size, dim)
    latest_fitness: Tensor  # (pop_size,)


def init(config: BestMonitorConfig, key: Tensor) -> BestMonitorState:
    """Allocate the fixed-shape buffers; the workflow has already filled pop_size/dim."""
    if config.pop_size is None or config.dim is None:
        raise ValueError(
            "BestMonitorConfig needs pop_size and dim; declare those fields and let "
            f"StdWorkflow fill them, got pop_size={config.pop_size}, dim={config.dim}"
        )
    return BestMonitorState(
        best_solution=_constant(np.zeros(config.dim)),
        best_fitness=_constant(np.inf),
        latest_solution=_constant(np.zeros((config.pop_size, config.dim))),
        latest_fitness=_constant(np.full((config.pop_size,), np.inf)),
    )


def monitor_update(
    config: BestMonitorConfig,
    state: BestMonitorState,
    candidate: Tensor,
    fitness: Tensor,
) -> BestMonitorState:
    """Track this generation's best candidate and the running best (min semantics)."""
    best_index = enp.argmin(fitness)
    gen_best_solution = evoxc.gather(candidate, best_index, axis=0)
    gen_best_fitness = enp.min(fitness)
    improved = gen_best_fitness < state.best_fitness
    return BestMonitorState(
        best_solution=evoxc.select(improved, gen_best_solution, state.best_solution),
        best_fitness=evoxc.select(improved, gen_best_fitness, state.best_fitness),
        latest_solution=candidate,
        latest_fitness=fitness,
    )


def record_auxiliary(config: BestMonitorConfig, aux: dict[str, Any]) -> None:
    """Host-side hook: append one generation of auxiliary tensors to ``aux_history``."""
    for name, value in aux.items():
        array = np.asarray(value.to(evoxc.core.Device("cpu")).numpy())
        config.aux_history.setdefault(name, []).append(array)


class Monitor:
    """Host-side wrapper created by ``StdWorkflow`` as ``Monitor(config=..., state=...)``."""

    def __init__(self, config: BestMonitorConfig, state: BestMonitorState):
        self.config = config
        self.state = state

    @staticmethod
    def _to_numpy(tensor: Tensor) -> np.ndarray:
        return np.asarray(tensor.to(evoxc.core.Device("cpu")).numpy())

    def get_best_fitness(self) -> np.ndarray:
        """Best-so-far fitness, with the opt-direction scaling undone."""
        scale = np.asarray(self.config.opt_direction, dtype=np.float32).reshape(())
        return self._to_numpy(self.state.best_fitness) * scale

    def get_best_solution(self) -> np.ndarray:
        """Best-so-far solution in search space."""
        return self._to_numpy(self.state.best_solution)

    @property
    def aux_history(self) -> dict:
        """Auxiliary tensors recorded through ``record_auxiliary``."""
        return self.config.aux_history
```

Run it with a built-in algorithm/problem pair, saved as `run_monitor.py` next to it:

```python
"""Run a built-in algorithm/problem pair with the custom monitor."""

import numpy as np

from evox_etl import StdWorkflow
from evox_etl.algorithms import PSO
from evox_etl.problems.numerical import Sphere
from my_monitor import BestMonitorConfig

dim = 5
workflow = StdWorkflow(
    PSO(pop_size=100, lb=tuple([-10.0] * dim), ub=tuple([10.0] * dim)),
    Sphere(),
    monitor=BestMonitorConfig(),
    opt_direction="min",
    num_generations=100,
)
workflow.run(seed=7)
print("best:", workflow.monitor.get_best_fitness())
print("solution norm:", np.linalg.norm(workflow.monitor.get_best_solution()))
print("aux keys:", list(workflow.monitor.aux_history))
```

Expected output: best fitness ≈ `1.2e-12`, solution norm ≈ `1e-6`, and `aux keys: []`.
The built-in PSO module defines no `record_step` hook, so the auxiliary channel stays
empty (documented behaviour); with an algorithm that defines it, keys appear per generation.

### Reading the example

- **Config.** `pop_size`/`dim` default to `None` and are filled by `StdWorkflow`; the
  workflow replaces `opt_direction`, while the host-side `aux_history` list survives.
- **`_constant`.** Traced code cannot capture host NumPy arrays, so `_constant` bakes
  them into in-graph constants; `best_fitness` starts at `+inf` so generation 0 improves.
- **`init`.** It refuses to guess: a missing `pop_size` or `dim` produces the explicit
  `ValueError` recommended above instead of a shape error deep in the graph.
- **`monitor_update`.** `enp.argmin` picks the generation's best row, `evoxc.gather`
  extracts it, and the two `evoxc.select` calls keep the running best — the scalar
  `improved` predicate implements min semantics without Python branching.
- **`record_auxiliary`.** It converts each concrete tensor to a NumPy array and appends
  it under its key, once per key per generation, with no `full_*_history` flag needed.
- **The wrapper.** `Monitor` stores config and state; `get_best_fitness` multiplies by
  `opt_direction` to undo the workflow's scaling (a `"max"` run reports the maximized
  value), `get_best_solution` returns the search-space point, and the workflow refreshes
  `wrapper.state` after every generation.

## Next

- {doc}`custom-alg-pro` — the algorithm side, including the `record_step` hook.
- {doc}`architecture` — where monitor state lives and how a generation is traced.
- {doc}`../advanced/viz` — plot the recorded history.
- {doc}`../../tutorial/tutorial_part3` — the built-in monitor plus `run()`/`fit()`.
