# 2. Your First Optimization: PSO on Sphere

This chapter takes you from an empty file to a finished optimization run.
You will build a particle swarm optimizer (PSO) that minimizes the 10-dimensional Sphere
function for 50 generations, then inspect the best solution it found, plot the
convergence curve, and swap the algorithm or the problem with a single line.

Everything here is the **EvoX 2.0 pre-release API** (import name `evox_etl`), the
functional rewrite of EvoX on the `evoxc` compiler. Components are **frozen config
dataclasses** plus plain step functions — there are no component classes, no
`Parameter`/`Mutable`, and no `torch.compile`.

## The complete program

Create `pso_sphere.py`:

```python
import numpy as np

from evox_etl import EvalMonitorConfig, StdWorkflow
from evox_etl.algorithms import PSO
from evox_etl.problems.numerical import Sphere

dim = 10
lb = np.full(dim, -100.0)
ub = np.full(dim, 100.0)

algorithm = PSO(pop_size=100, lb=lb, ub=ub)   # w=0.6, phi_p=2.5, phi_g=0.8
problem = Sphere()
monitor = EvalMonitorConfig()

wf = StdWorkflow(
    algorithm=algorithm,
    problem=problem,
    monitor=monitor,
    opt_direction="min",
    num_generations=50,
    backend="numpy",
)

state = wf.run(seed=42)

print(f"generations  : {len(wf.monitor.fitness_history)}")
print(f"best fitness : {wf.monitor.get_best_fitness():.6f}")
print(f"best solution: {np.round(wf.monitor.get_best_solution(), 4)}")
print(f"||x*||       : {np.linalg.norm(wf.monitor.get_best_solution()):.6f}")
```

Running it prints (results are deterministic for a fixed seed):

```text
generations  : 50
best fitness : 0.027914
best solution: [ 0.0316  0.0206  0.0153  0.0919 -0.034  -0.0754  0.0415 -0.0136  0.0088
 -0.0948]
||x*||       : 0.167076
```

The Sphere function is minimized at the origin with fitness `0`, so after 50
generations the swarm is within `0.17` of the optimum in L2 norm. The whole run
traces a handful of compiled graphs and executes 50 generation steps in well under a
second on the NumPy backend — the same code on a GPU backend is covered at the end of this
chapter.

```{note}
`PSO(pop_size, lb, ub, w=0.6, phi_p=2.5, phi_g=0.8)` has **no `dim` argument**: the
problem dimension is simply the length of `lb`/`ub`. The bounds are the only place
the search space is described, and they must be 1-D and of equal length.
```

## What each argument means

`StdWorkflow` is the only object you construct that is not a config dataclass.
Every other argument is either a frozen dataclass or a plain value:

| Argument | Meaning | Example |
|---|---|---|
| `algorithm=` | Config dataclass of an algorithm. Its module must define `step(config, state, evaluate)` (optionally `init_step`/`final_step`). | `PSO(pop_size=100, lb=lb, ub=ub)` |
| `problem=` | Config dataclass of a problem. Its module defines `evaluate(config, state, x)` and returns `(fitness, new_state)`. | `Sphere()` |
| `monitor=` | Config dataclass of a monitor, or `None` for no monitoring at all. | `EvalMonitorConfig()` |
| `opt_direction=` | `"min"` (default) or `"max"`; a **list** of them for multi-objective problems. | `"min"` |
| `num_generations=` | Default generation budget used by `run()`/`fit()` when they are called without a count. | `50` |
| `backend=` | The `evoxc` compiler backend that executes the step graph: `"numpy"`, or an IREE/TVM/XLA backend id. | `"numpy"` |
| `device=` | Device the step graph runs on: `None` (CPU), `"cpu"`, `"cuda"`, `"cuda:1"`, or an `evoxc.Device`. | `None` |
| `compile_options=` | Extra keyword options forwarded to `evoxc.build` (e.g. an optimization level or a cache directory). `backend`/`device` keys are ignored here. | `None` |

The constructor also accepts `solution_transform=` and `fitness_transform=`; both are
explained in {doc}`Part 3 <tutorial_part3>`.

Two of these are pure plumbing you will rarely touch: `backend=` picks the compiler
that turns the step graph into an executable, and `device=` picks where the resulting
program's tensors live.

## Driving the loop yourself

`wf.run(seed=42)` is a convenience wrapper. Underneath it does exactly this, and you
can write the same thing yourself whenever you need to interleave other work with
generations (log, checkpoint, adapt a hyperparameter, stop early):

```python
generations = 50

state = wf.init(seed=42)          # build the graphs, return the generation-0 state
state = wf.init_step(state)       # generation 1
for _ in range(generations - 2):  # generations 2 .. 49
    state = wf.step(state)
state = wf.final_step(state)      # generation 50
wf.flush()                        # drain the monitor readback of the last generation
```

Two details matter:

- `step()` **never** switches to `init_step` or `final_step` on its own — the
  dispatch is decided host-side, by you (or by `run()`). And when the algorithm
  module does not define `init_step`/`final_step`, the workflow silently falls back
  to ordinary `step` for those two calls.
- Monitor history entries are copied device→host one generation *behind* the step.
  `run()`/`fit()` flush that pipeline when their loop ends; when you drive the loop
  manually you must call `wf.flush()` before reading `fitness_history` or any other
  history property, otherwise the last generation is missing from the list.

On a fresh workflow the loop above reproduces `run(50)` exactly: the same best fitness
`0.027914` and the same 50 history entries. Re-stepping an *already run* workflow is
legal but appends to the same history lists, so the entries concatenate — build a new
`StdWorkflow` (and a new `EvalMonitorConfig`) for an independent run.

## Reading results

`wf.monitor` is a host-side wrapper over the monitor state; the raw tensor state is
`wf.monitor_state`:

```python
best = wf.monitor.get_best_fitness()        # Python float
x_best = wf.monitor.get_best_solution()     # numpy.ndarray of shape (dim,)
history = wf.monitor.fitness_history        # list of numpy arrays, one per generation
```

- `get_best_fitness()` returns a `float` (not a tensor) — you can format or compare
  it directly, as the program above does.
- `get_best_solution()` returns a 1-D `numpy.ndarray` of length `dim`, already
  copied to the host.
- `fitness_history` (alias of `fit_history`) holds the **whole** generation's fitness
  vector for every generation, so after a 50-generation run it is a list of 50 arrays
  (not 50 scalars).

## Plotting

`EvalMonitorConfig()` records the fitness history by default, so plotting needs one
call:

```python
fig = wf.monitor.plot()
fig.show()          # or fig.write_html("convergence.html")
```

`plot()` returns a `plotly.graph_objects.Figure` — Plotly ships with the optional
visualization extra (`pip install "evox[vis]"`). Without it, `plot()` warns and
returns `None` instead of raising, and so does a call on an empty history.

For multi-objective runs, pass the problem's known Pareto front as a
`(k, n_obj)` NumPy array to overlay it: `wf.monitor.plot(problem_pf=pf_array)`
(see {doc}`Part 6 <tutorial_part6>`).
The figure switches from a 1-D curve to a 2-D/3-D objective-space scatter depending
on the number of objectives.

## One-line swaps

The algorithm is just a config dataclass, so replacing PSO is a one-line change —
here for `DE` (differential evolution) and CMA-ES:

```python
from evox_etl.algorithms import DE, make_cma_es

# differential evolution — same bounds, same everything else
algorithm = DE(pop_size=100, lb=lb, ub=ub)

# CMA-ES: it is parameterized by a mean and a step size, NOT by bounds
algorithm = make_cma_es(mean_init=np.zeros(dim), sigma=20.0, pop_size=20)
monitor = EvalMonitorConfig(dim=dim)   # CMA-ES cannot advertise `dim` to the monitor
```

Best fitness on the 10-D Sphere above (box `[-100, 100]`, seed 42):

| Algorithm | 50 generations | 200 generations |
|---|---|---|
| `PSO(pop_size=100, lb=lb, ub=ub)` | 0.0279 | 6.6e-19 |
| `DE(pop_size=100, lb=lb, ub=ub)` | 72.7 | 2.5e-05 |
| `make_cma_es(mean_init=np.zeros(dim), sigma=20.0, pop_size=20)` | 0.107 | 1.1e-15 |

DE's default hyperparameters (`rand/1`, `differential_weight=0.5`,
`cross_probability=0.9`) need more than 50 generations on a box this wide — that is a
property of the algorithm, not of the workflow. CMA-ES is built around `mean_init`
and `sigma` rather than `lb`/`ub`, and its config carries no `dim` field, so the
workflow cannot infer the solution dimension for the monitor: pass `dim=` (or
`EvalMonitorConfig(dim=..., pop_size=...)`) yourself.

The problem is equally easy to swap — `Ackley()` and `Rastrigin()` are drop-in
replacements for `Sphere()`, with the same `evaluate` contract:

```python
from evox_etl.problems.numerical import Ackley, Rastrigin

problem = Ackley()       # or Rastrigin()
```

Best fitness of PSO (`pop_size=100`) on a narrower box `[-5, 5]^10`, seed 42, where
these landscapes are actually reachable:

| Problem | 50 generations | 200 generations |
|---|---|---|
| `Sphere()` | 7.0e-05 | 1.7e-21 |
| `Ackley()` | 0.0102 | 9.5e-07 |
| `Rastrigin()` | 16.0 | 1.99 |

Rastrigin stays hard for a basic PSO — it is a massively multimodal landscape and a
single swarm with fixed inertia tends to settle in a local basin. That is the point
of the rest of this tutorial: once the plumbing is in place, the algorithm becomes a
one-line experiment.

```{tip}
Switching to a different compiler is only a `backend=` change — for example an
IREE/TVM/XLA backend instead of the NumPy one — with the algorithm, problem and
monitor untouched. See {doc}`Advanced: choosing a backend <../guide/advanced/backends>` for
the available backend ids and what each of them supports.
```

```{note}
The same program runs unchanged on CUDA-capable backends: pass `device="cuda"` (and a
GPU backend id) and the workflow places the state and the step graph on the device,
while the monitor keeps returning plain host-side NumPy arrays.
```
