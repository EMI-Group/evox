# Single-objective optimization: PSO on Sphere

This page builds the smallest useful EvoX 2.0 program: a particle swarm optimizer (PSO) minimizing
the 10-dimensional Sphere function.
It then grows that program three ways — a measured comparison against DE and CMA-ES, a convergence
plot, and a different benchmark problem — without touching the workflow code.

```{warning} Experimental EvoX 2.0 pre-release
The code below uses the `evox_etl` import name (the functional rewrite on the `evoxc` compiler) and
the **NumPy backend** (`backend="numpy"`), so it runs on any machine without a compiler toolchain.
The 1.x torch API (`import evox`, `Mutable`, `torch.compile`) is not used.
See {doc}`../tutorial/tutorial_part2` for the same program explained line by line, and
{doc}`../guide/advanced/backends` for switching the backend to IREE, TVM or XLA devices.
```

## The complete program

Save this as `pso_sphere.py` (it mirrors `examples/quickstart.py` in the repository) and run it with
`PYTHONPATH=src python pso_sphere.py`:

```python
import numpy as np
from evox_etl import StdWorkflow, EvalMonitorConfig
from evox_etl.algorithms import PSO
from evox_etl.problems.numerical import Sphere

dim = 10
lb = np.full(dim, -100.0)
ub = np.full(dim, 100.0)

algorithm = PSO(pop_size=100, lb=lb, ub=ub)   # w=0.6, phi_p=2.5, phi_g=0.8
problem = Sphere()                            # minimum at x = [0, ..., 0], fitness 0
monitor = EvalMonitorConfig()                 # full fitness history, top-1 elite

workflow = StdWorkflow(
    algorithm=algorithm,
    problem=problem,
    monitor=monitor,
    opt_direction="min",
    num_generations=50,
    backend="numpy",
)

state = workflow.init(seed=42)
for generation in range(1, 51):
    state = workflow.step(state)
    if generation % 10 == 0:
        best = float(np.asarray(workflow.monitor.get_best_fitness()).reshape(-1)[0])
        print(f"  generation {generation:>2}: best fitness = {best:.6f}")

best = float(np.asarray(workflow.monitor.get_best_fitness()).reshape(-1)[0])
best_solution = np.asarray(workflow.monitor.get_best_solution())
print(f"\nFinal best fitness: {best:.6f}")
print(f"Best solution L2 norm: {np.linalg.norm(best_solution):.6f}")
```

The run is deterministic for a fixed seed, so it prints exactly:

```text
  generation 10: best fitness = 1322.229492
  generation 20: best fitness = 34.088341
  generation 30: best fitness = 3.121520
  generation 40: best fitness = 0.098329
  generation 50: best fitness = 0.007325

Final best fitness: 0.007325
Best solution L2 norm: 0.085587
```

The true optimum is the origin with fitness `0`, so after 50 generations the swarm sits within
`0.086` of it in L2 norm. The monitor, not the algorithm, owns the result: `get_best_fitness()`
returns the best value seen over every generation and `get_best_solution()` the matching solution
vector.

The loop is deliberately the *hand-driven* variant: it calls `step` 50 times on the raw `init` state
and never dispatches `init_step`, `final_step` or `flush()`.
That is why it does not reproduce `workflow.run(50, seed=42)` — it follows a different trajectory, not
the `run()` trajectory shifted by one generation (see the tip under *The objects you construct* below).

```{note}
`PSO(pop_size, lb, ub, w=0.6, phi_p=2.5, phi_g=0.8)` has **no `dim` argument** — the dimension is
the length of `lb`/`ub`. In this page the bounds are NumPy arrays, which the dataclass stores as
given; the `make_pso(pop_size, lb, ub, ...)` constructor in `evox_etl.algorithms` validates them
and normalizes them to flat Python float tuples, which is what the comparison below uses.
```

## The objects you construct

Everything except `StdWorkflow` is a **frozen config dataclass**, and the algorithm/problem configs are
never mutated during the run — the only thing that grows is the monitor config's history lists,
which the workflow fills host-side after each generation.

| Object | Type | Role |
|---|---|---|
| `PSO(...)` | algorithm config | Its module defines `init(config, key)` and `step(config, state, evaluate)`; `step` owns one full generation. |
| `Sphere()` | problem config | Its module defines `evaluate(config, state, x) -> (fitness, new_state)`; numerical problems are stateless. |
| `EvalMonitorConfig()` | monitor config | Records the fitness history host-side and gives you the `get_best_*` accessors. |
| `StdWorkflow(...)` | plain class | Traces the three configs into one step graph per step variant and runs the loop. |

`workflow.init(seed=42)` traces and runs the *initialization* graph once and returns a
`WorkflowState` pytree of evoxc tensors; `workflow.step(state)` runs one compiled generation and
returns the next state.
There is no global seed — the key lives inside the state, which is why `seed=` is passed to
`init`/`run` and never to the components.

```{tip}
`workflow.run(50, seed=42)` is the shortcut for the loop above, but it is not the same loop.
`run(generations)` dispatches generation 1 to the algorithm's `init_step` (evaluate the initial
population, do not move yet) and the last generation to `final_step`, then drains the monitor
readbacks — so `run(50)` is `init` + `init_step` + 49 × `step` (the 49th `step` being the
`final_step` dispatch, which falls back to `step` because PSO defines no `final_step`).
The loop above is instead `init` + 50 × `step`, and its first `step` runs on the raw `init` state,
whose fitness vector is still `inf`: the cognitive term is zero (the local best is still the current
positions) and the social best is particle 0, the tie-break when every fitness is `inf`.
The two runs therefore diverge from that first move instead of being one generation apart —
`run(50, seed=42)` reaches `0.027914`, the loop above reaches `0.007325` (its 49th generation is
`0.010226`), and `run(51, seed=42)` reaches `0.027468`, which is still not the loop's 50-generation
value.
To follow the `run()` trajectory by hand, call `init_step` before the loop and `final_step` after it
(both fall back to `step` when the algorithm has no such variant), and call `flush()` before reading
the monitor on a device backend.
```

## Comparing PSO, DE and CMA-ES

Only the algorithm config changes. The `make_*` constructors are the validating front doors
(`make_pso`, `make_de`, `make_cma_es`), and all three are run through the identical workflow:

```python
import numpy as np
from evox_etl import StdWorkflow, EvalMonitorConfig
from evox_etl.algorithms import make_pso, make_de, make_cma_es
from evox_etl.problems.numerical import Sphere

dim = 10
lb, ub = np.full(dim, -100.0), np.full(dim, 100.0)

algorithms = {
    "PSO": make_pso(pop_size=100, lb=lb, ub=ub),
    "DE": make_de(pop_size=100, lb=lb, ub=ub),                      # F=0.5, CR=0.9, "rand/1"
    "CMA-ES": make_cma_es(mean_init=np.zeros(dim), sigma=10.0, pop_size=100),
}

for name, config in algorithms.items():
    workflow = StdWorkflow(
        config, Sphere(),
        monitor=EvalMonitorConfig(dim=dim),
        opt_direction="min",
        num_generations=200,
        backend="numpy",
    )
    workflow.run(seed=42)
    print(f"{name:7s} best = {workflow.monitor.get_best_fitness():.6e}")
```

Best fitness on Sphere (10-D, bounds `[-100, 100]`, population 100, **one run per cell, seed 42**);
the left column is the same program with `num_generations=50`:

| Algorithm | best after 50 generations | best after 200 generations |
|---|---|---|
| PSO (`make_pso`) | `2.791427e-02` | `6.569471e-19` |
| DE (`make_de`) | `7.271989e+01` | `2.477526e-05` |
| CMA-ES (`make_cma_es`) | `3.903870e-05` | `4.585538e-19` |

Read the table for shape, not for ranking: 200 generations are enough for PSO and CMA-ES to reach
machine precision on this easy problem, DE lags at `2.5e-05`, and a single seed says nothing about
robustness. DE is the slowest starter here because `rand/1` with `F=0.5` contracts the population
gradually from a search box of width 200.

```{note}
`EvalMonitorConfig(dim=dim)` passes the solution dimension explicitly, which is only needed for
algorithms whose state carries no `population`/`pop` tensor — CMA-ES keeps a `mean` and a
covariance matrix instead. Without it the workflow cannot infer `dim` and fails during `init` with
`EvalMonitor requires pop_size and dim to initialize its buffers, got pop_size=100, dim=None. The
workflow fills them from the algorithm state (population/pop) or the algorithm config.`
For PSO/DE the plain `EvalMonitorConfig()` is enough, so you can drop the argument there.
```

## Plotting the convergence

The monitor keeps one NumPy array of fitness values per generation, so plotting needs no extra
plumbing:

```python
# `workflow` is still the last (CMA-ES) run of the loop above
figure = workflow.monitor.plot()                 # animated, one frame per generation
static = workflow.monitor.plot(animation=False)  # static Min / Max / Median / Average lines
```

`plot()` returns a `plotly.graph_objects.Figure` — four traces (`Min`, `Max`, `Median`, `Average`
of the population fitness), plus one frame per recorded generation when `animation=True` (the
default). Display it with `figure.show()`, or embed it in a notebook as the last expression of a
cell.

```{tip}
`workflow.monitor.get_fitness_history()` returns the per-generation NumPy arrays behind the figure
if you want to build your own plot; they are already un-negated, so under `opt_direction="max"` the
recorded values still read "larger is better".
The full builder list — 1-D, 2-D and 3-D objective-space figures, plus the `.exv` serializer — is
documented in {doc}`../guide/advanced/viz`.
```

## Changing the problem

`StdWorkflow` takes the problem as data, so swapping `Sphere()` for another benchmark function is a
one-line change. The search box lives in the *algorithm* config, so it does not follow the problem:
here all three functions are run in the wide symmetric `[-32.768, 32.768]` box (Ackley's conventional
domain; Rastrigin is usually benchmarked in `[-5.12, 5.12]`, so this is the harder variant):

```python
import numpy as np
from evox_etl import StdWorkflow, EvalMonitorConfig
from evox_etl.algorithms import make_pso
from evox_etl.problems.numerical import Sphere, Ackley, Rastrigin

dim = 10
lb, ub = np.full(dim, -32.768), np.full(dim, 32.768)

for problem in (Sphere(), Ackley(), Rastrigin()):
    workflow = StdWorkflow(
        make_pso(pop_size=100, lb=lb, ub=ub), problem,
        monitor=EvalMonitorConfig(),
        opt_direction="min",
        num_generations=200,
        backend="numpy",
    )
    workflow.run(seed=42)
    print(f"{type(problem).__name__:10s} best = {workflow.monitor.get_best_fitness():.6e}")
```

```text
Sphere     best = 2.802604e-19
Ackley     best = 9.536743e-07
Rastrigin  best = 4.166534e+00
```

PSO drives Sphere to `2.8e-19`, gets Ackley down to `9.5e-7` — its local minima are shallow enough for
a global-best swarm to slide into the global basin — and stalls on Rastrigin at `4.2`, a lattice of
local minima on a unit grid where a plain global-best PSO has no mechanism to escape once the swarm
has contracted. That is a property of the algorithm, not of the API: swapping in another PSO variant
from `evox_etl.algorithms` is again a one-line change.

The other built-in numerical problems live in `evox_etl.problems.numerical`:
`Griewank`, `Rosenbrock`, `Schwefel`, `Ellipsoid`, `Zakharov`, `Levy`, the `DTLZ1`–`DTLZ7` family
(multi-objective, see {doc}`mo`) and `CEC2022`.

## Where to go next

- Multi-objective optimization with NSGA-II, Pareto fronts, metrics and `.exv` export: {doc}`mo`.
- Training a neural network with the training-free virtual ES family: {doc}`neuroevolution`.
- Write your own algorithm or problem config: {doc}`../guide/developer/custom-alg-pro`.
- Hyperparameter optimization over a workflow: {doc}`../guide/advanced/hpo`.
