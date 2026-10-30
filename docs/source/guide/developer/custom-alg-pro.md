# Custom algorithms and problems in EvoX

This page contains two worked examples: first a custom algorithm run against a built-in
problem, then a custom problem run with a built-in algorithm.

Every component follows the same shape: a **frozen config dataclass** holding static
hyperparameters, plus **plain module-level functions** (`init`, `init_step`, `step`,
or `evaluate`) in the same module. `StdWorkflow` receives the config object and locates
the functions with `importlib.import_module(type(config).__module__)` — the module a
config was defined in *is* the component.

```{warning}
Put exactly one component in each module. Two components in one file live in the same
module, so the workflow would resolve the same shared names (`init`, `step`, `evaluate`)
for both, and the two components would silently collide. Give every algorithm, problem,
and monitor its own file.
```

```{note}
This page shows only the component contract. How `StdWorkflow` validates configs,
traces a generation, and drives the state is described in {doc}`architecture`, and a
full end-to-end walkthrough is in {doc}`../../tutorial/tutorial_part3`.
```

## Part 1 — Write your own algorithm

The algorithm module, `my_algorithm.py`, consists of six pieces:

- `GaussianES` — a frozen dataclass holding the hyperparameters (`pop_size`, `dim`,
  `sigma`, `lb`, `ub`). Its leaves are plain Python values and stay static for the whole
  run.
- `make_gaussian_es(...)` — the host-side entry point following the `make_*` convention:
  it validates and normalises the arguments (checking ranges, casting to `int`/`float`)
  and returns the config. Doing this host-side keeps the traced code free of checks.
- `GaussianESState` — a dataclass whose leaves are `evoxc` tensors only, each with a
  fixed shape, including the RNG `key`.
- `init(config, key)` — samples the initial population uniformly inside the box with
  `random.uniform`.
- `init_step(config, state, evaluate)` — generation 0. Instead of mutating the initial
  population it evaluates it directly and seeds the best trackers, so no generation is
  wasted.
- `step(config, state, evaluate)` — owns exactly one full generation: it splits
  `state.key`, mutates the population, clamps the trial with `enp.clip`, calls
  `fitness = evaluate(trial)`, keeps the better of parent and child with `evoxc.select`,
  and advances the key.

```{note}
`evaluate` is an opaque workflow closure, not a user function. It accepts a batch of
candidates and returns fitness already scaled to **minimisation**, and your `step` may
call it more than once per generation (for example with trial batches or a restart). Treat
it as a black box that returns a tensor of shape `(pop_size,)`.
```

The complete `my_algorithm.py`:

```python
"""A minimal evolution strategy written as an evox algorithm module."""

from __future__ import annotations

import dataclasses
from typing import Any

import evoxc
import evoxc.numpy as enp
import evoxc.random as random
import numpy as np

Tensor = evoxc.SymbolicTensor
F32 = np.dtype("float32")


@dataclasses.dataclass(frozen=True)
class GaussianES:
    """Hyperparameters: plain Python values, fixed for the whole run."""

    pop_size: int
    dim: int
    sigma: float = 0.1
    lb: float = -10.0
    ub: float = 10.0


def make_gaussian_es(
    pop_size: int, dim: int, sigma: float = 0.1, lb: float = -10.0, ub: float = 10.0
) -> GaussianES:
    """Validate the hyperparameters and return a config with plain Python leaves."""
    if pop_size <= 0:
        raise ValueError(f"pop_size must be positive, got {pop_size}")
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    if sigma <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if not lb < ub:
        raise ValueError(f"require lb < ub, got lb={lb}, ub={ub}")
    return GaussianES(
        pop_size=int(pop_size), dim=int(dim), sigma=float(sigma), lb=float(lb), ub=float(ub)
    )


@dataclasses.dataclass(frozen=True)
class GaussianESState:
    """Algorithm state: evoxc tensor leaves only."""

    population: Tensor  # (pop_size, dim)
    fitness: Tensor  # (pop_size,)
    best_solution: Tensor  # (dim,)
    best_fitness: Tensor  # ()
    key: Tensor  # ()


def init(config: GaussianES, key: Tensor) -> GaussianESState:
    """Sample the initial population uniformly inside the box."""
    key, subkey = random.split(key)
    population = random.uniform(
        subkey, (config.pop_size, config.dim), config.lb, config.ub, F32
    )
    return GaussianESState(
        population=population,
        fitness=enp.full((config.pop_size,), np.inf, dtype=F32),
        best_solution=population[0],
        best_fitness=enp.full((), np.inf, dtype=F32),
        key=key,
    )


def init_step(
    config: GaussianES, state: GaussianESState, evaluate: Any
) -> GaussianESState:
    """Generation 0: evaluate the initial population instead of mutating it."""
    fitness = evaluate(state.population)
    best_index = enp.argmin(fitness)
    return dataclasses.replace(
        state,
        fitness=fitness,
        best_solution=evoxc.gather(state.population, best_index, axis=0),
        best_fitness=enp.min(fitness),
    )


def step(config: GaussianES, state: GaussianESState, evaluate: Any) -> GaussianESState:
    """One generation: mutate, evaluate, keep the better of parent and child."""
    key, subkey = random.split(state.key)
    noise = random.normal(subkey, (config.pop_size, config.dim), dtype=F32)
    trial = enp.clip(state.population + config.sigma * noise, config.lb, config.ub)
    trial_fitness = evaluate(trial)
    improved = trial_fitness < state.fitness
    population = evoxc.select(
        enp.expand_dims(improved, axis=1), trial, state.population
    )
    fitness = evoxc.select(improved, trial_fitness, state.fitness)
    best_index = enp.argmin(fitness)
    return dataclasses.replace(
        state,
        population=population,
        fitness=fitness,
        best_solution=evoxc.gather(population, best_index, axis=0),
        best_fitness=enp.min(fitness),
        key=key,
    )
```

With the algorithm in place, the runner only has to plug it into a workflow with a
built-in problem and a monitor:

```python
"""Run the custom algorithm with a built-in problem and monitor."""

import numpy as np

from evox import EvalMonitorConfig, StdWorkflow
from evox.problems.numerical import Sphere
from my_algorithm import make_gaussian_es

workflow = StdWorkflow(
    make_gaussian_es(pop_size=64, dim=5, sigma=0.2),
    Sphere(),
    monitor=EvalMonitorConfig(),
    opt_direction="min",
    num_generations=200,
)
best = workflow.fit(generations=200, seed=0)
print("best fitness:", float(np.asarray(best).reshape(-1)[0]))
```

For seed `0` this prints a small value around `0.0124`:

```
best fitness: 0.0124...
```

That is roughly `0.01`, **not exactly zero**: this is a simple mutation-only strategy,
not a tuned optimiser. The exact digits depend on the backend and on how many
generations you run.

```{tip}
`workflow.fit(...)` returns the best-so-far value and also records history on the
monitor. Call `fit` with a fixed `seed` while you iterate on the algorithm so that
changes in the printed value reflect your code, not noise.
```

## Part 2 — Write your own problem

The problem module, `my_problem.py`, mirrors the same layout with an `evaluate` function
instead of a step function:

- `ShiftedSphere` — the frozen config. `center` is a flat tuple of Python floats, which
  makes it a static leaf; `make_shifted_sphere` produces it from any array-like input.
- `evaluate(config, state, pop)` — returns `(fitness, state)`. Fitness has shape `(n,)`
  for a single-objective problem or `(n, n_obj)` for a multi-objective one. This problem
  is stateless, so it returns the incoming state unchanged (the workflow passes
  `EmptyState` when the module defines no `init`).
- Host arrays inside a trace must be baked once with
  `evoxc.ops.constant(evoxc.core.tensor(np.asarray(...)))`; never put NumPy arrays in the
  config, where they would break static-leaf rules.
- A problem may keep state — a counter or a cache, for instance — by declaring a state
  dataclass and an `init(config, key)` function in the same module. In that case
  `evaluate` receives the current state and must return the updated one.

```{warning}
The `(fitness, state)` return value is not optional, even for a problem that carries no
state. The workflow always unpacks the pair, so returning only the fitness fails
immediately.
```

The complete `my_problem.py`:

```python
"""A minimal shifted-sphere problem written as an evox problem module."""

from __future__ import annotations

import dataclasses
import math

import evoxc
import evoxc.numpy as enp
import numpy as np


@dataclasses.dataclass(frozen=True)
class ShiftedSphere:
    """The sphere function minimised at ``center``; the center is a static leaf."""

    center: tuple[float, ...]
    scale: float = 1.0


def make_shifted_sphere(center, scale: float = 1.0) -> ShiftedSphere:
    """Normalise ``center`` to a flat tuple of Python floats and validate ``scale``."""
    center_array = np.asarray(center, dtype=np.float64)
    if center_array.ndim != 1:
        raise ValueError(f"center must be 1-D, got shape {center_array.shape}")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"scale must be positive and finite, got {scale}")
    return ShiftedSphere(
        center=tuple(float(v) for v in center_array), scale=float(scale)
    )


def evaluate(config: ShiftedSphere, state, pop: evoxc.SymbolicTensor):
    """Return ``(fitness, state)``; this problem is stateless, so state round-trips."""
    center = evoxc.ops.constant(
        evoxc.core.tensor(np.asarray(config.center, dtype=np.float32))
    )
    diff = pop - center
    fitness = config.scale * enp.sum(diff * diff, axis=1)
    return fitness, state
```

The runner pairs the custom problem with a built-in algorithm:

```python
"""Run a built-in algorithm against the custom problem."""

import numpy as np

from evox import EvalMonitorConfig, StdWorkflow
from evox.algorithms import PSO
from my_problem import make_shifted_sphere

dim = 5
problem = make_shifted_sphere(center=[1.0, -2.0, 0.5, 2.0, 0.0])
workflow = StdWorkflow(
    PSO(pop_size=100, lb=tuple([-10.0] * dim), ub=tuple([10.0] * dim)),
    problem,
    monitor=EvalMonitorConfig(),
    opt_direction="min",
    num_generations=100,
)
best = workflow.fit(generations=100, seed=0)
print("best fitness:", float(np.asarray(best).reshape(-1)[0]))
print("distance to the optimum:", np.linalg.norm(workflow.monitor.get_best_solution() - np.asarray(problem.center)))
```

The expected output for seed `0` is a best fitness near `1e-13` and a distance to the
optimum near `1e-6`: PSO drives both the objective and the distance to the center toward
zero, and the remaining gap is the usual numerical residue of a finite run.

## Part 3 — Keeping component code traceable

The workflow records a trace once and then executes that graph, so component code has to
obey a few rules:

- All tensor work must use `evoxc` / `evoxc.numpy` (`enp`) / `evoxc.random` ops. There is
  no eager mode, so calling those ops outside a trace raises the `evoxc` `TraceError`.
- No Python control flow on tensor values. `if fitness < best:` is illegal; use
  `evoxc.select`, `evoxc.where`, or a reduction instead.
- Shapes must be static. The workflow traces one graph per state-leaf signature, so a
  state whose shapes change between generations forces a re-trace. `pop_size` and `dim`
  belong in the config, not in the state.
- Config values are static. Reading `config.sigma` inside a traced function is fine, but
  a config value cannot be a tensor; normalise arrays host-side in `make_*`.
- Host-side recursion in Python — plain `for` loops over static counts — runs at trace
  time and is fine; use `evoxc.scan` / `evoxc.while_loop` only for graph-level loops.
- The algorithm always minimises. `evaluate` hands back opt-direction-scaled fitness, so
  `step` must treat smaller as better even when the user asked for maximisation.

## Part 4 — Multi-objective custom operators

`NSGA2Config` (and the other multi-objective configs) accept optional operator callables:
`selection_op`, `crossover_op`, and `mutation_op`. Their signatures must match the
defaults they replace:

| Callable | Signature |
|---|---|
| `selection_op` | `selection_op(key, n_parents, fitness)` |
| `crossover_op` | `crossover_op(key, x)` |
| `mutation_op` | `mutation_op(key, x, lb, ub)` |

Passing `None` selects the built-in operator. Because callables are not valid static
pytree leaves, a config that carries them must be registered with
`evoxc.register_pytree_node`; the multi-objective modules do this themselves at import
time (see `evox.algorithms.mo.nsga2`).

For the operator catalog, see {doc}`../advanced/operators`. For tuning algorithm
hyperparameters, see {doc}`../advanced/hpo`; neither catalog is duplicated here.

## Next

- {doc}`custom-monitor` — monitor hooks, reading a run back on the host, and history.
- {doc}`architecture` — how the workflow builds and executes the graph.
- {doc}`../../tutorial/tutorial_part3` — end-to-end workflow mechanics.
