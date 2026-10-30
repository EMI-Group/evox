# Multi-objective optimization: NSGA-II on DTLZ2

This page runs a multi-objective evolutionary algorithm (NSGA-II) on the 12-dimensional, two-objective
DTLZ2 problem, reads the achieved Pareto front out of the monitor, scores it with the built-in
IGD/GD/hypervolume metrics, plots it against the true front, and streams the whole run into an
`.exv` file.

```{warning} Experimental EvoX 2.0 pre-release
The code below uses the `evox_etl` import name (the functional rewrite on the `evoxc` compiler) and
the **NumPy backend** (`backend="numpy"`). The 1.x torch API — including `evox.metrics` taking plain
tensors and `EvalMonitor.plot` without a problem front — is not used.
```

## The complete program

```python
import numpy as np
from evox_etl import StdWorkflow, EvalMonitorConfig
from evox_etl.algorithms import make_nsga2
from evox_etl.problems.numerical import DTLZ2

d = 12
problem = DTLZ2(d=d, m=2)                                  # 2 objectives, d = 12 decision variables
algorithm = make_nsga2(pop_size=100, n_objs=2, lb=np.zeros(d), ub=np.ones(d))
workflow = StdWorkflow(
    algorithm=algorithm,
    problem=problem,
    monitor=EvalMonitorConfig(full_sol_history=True),      # keep solutions too, not just fitness
    opt_direction=["min", "min"],                         # one entry per objective
    num_generations=50,
    backend="numpy",
)
workflow.run(seed=42)

solutions, fitness = workflow.monitor.get_pf()
print(f"achieved front: {fitness.shape[0]} points, {fitness.shape[1]} objectives")
print(f"objective 0 range: {fitness[:, 0].min():.6f} .. {fitness[:, 0].max():.6f}")
print(f"objective 1 range: {fitness[:, 1].min():.6f} .. {fitness[:, 1].max():.6f}")
```

Output (deterministic for `seed=42`):

```text
achieved front: 335 points, 2 objectives
objective 0 range: 0.000000 .. 1.000913
objective 1 range: 0.000000 .. 1.001892
```

Three things differ from a single-objective run:

| Single-objective | Multi-objective |
|---|---|
| `opt_direction="min"` | `opt_direction=["min", "min"]` — one direction per objective, which is what puts the monitor in Pareto mode |
| `EvalMonitorConfig()` | `EvalMonitorConfig(full_sol_history=True)` — the front needs solutions as well as fitness values |
| `get_best_fitness()` / `get_best_solution()` | `get_pf()` (solutions **and** fitness) or `get_pf_fitness()` (fitness only) |

```{note}
The maximum of each objective slightly exceeds `1.0` because the 50-generation budget leaves part of
the population off the DTLZ2 sphere (the true front is the positive octant of the unit sphere,
`sum(f_i^2) = 1`). Non-dominated points just outside it are still non-dominated — a front is a set
of trade-offs, not a guarantee of optimality.
```

## Reading the Pareto front

`get_pf()` collects **every** evaluated solution (50 generations × 100 individuals), drops duplicates,
keeps the non-dominated ones and un-negates the fitness, so you never merge `sol_history` and
`fit_history` yourself:

```python
solutions, fitness = workflow.monitor.get_pf()          # (n, d) and (n, m) NumPy arrays
front_fitness = workflow.monitor.get_pf_fitness()      # fitness only, same non-dominated filter
```

Note the argument order: `get_pf()` returns `(solutions, fitness)`.
Pass `deduplicate=False` to keep repeated solutions.
The monitored fitness values are un-negated with the workflow's `opt_direction`, so under
`["max", "max"]` you would get the original maximization values back, not the internal minimization
form.

```{tip}
`get_pf_fitness()` already returns a `float32` array, so it can go straight into a metric call below
— no dtype conversion needed.
```

## Scoring the front: IGD, GD and hypervolume

The metric functions in `evox_etl.metrics` are **traced graph functions**, not host-side helpers:
there is no eager mode, so you build a one-shot executable with `evoxc.build` and run it with
`evoxc.run`. Static scalars (`num_sample`) must be passed to `evoxc.build` *and* re-passed by value
to `evoxc.run`.

```python
import numpy as np
import evoxc
from evoxc.core import TensorSpec
from evox_etl import key
from evox_etl.problems.numerical.dtlz import pf as dtlz_pf
from evox_etl.metrics import gd, igd, hv

front = evoxc.run(evoxc.build(dtlz_pf, problem), problem).numpy()   # (k, 2) true front
objs = np.asarray(fitness, dtype=np.float32)
print("true front:", front.shape)

spec_objs = TensorSpec(objs.shape, np.float32)
spec_front = TensorSpec(front.shape, np.float32)
igd_exe = evoxc.build(igd, spec_objs, spec_front, backend="numpy")
gd_exe = evoxc.build(gd, spec_objs, spec_front, backend="numpy")
print(f"IGD: {float(evoxc.run(igd_exe, objs, front).numpy()):.6f}")
print(f"GD : {float(evoxc.run(gd_exe, objs, front).numpy()):.6f}")

ref = (1.1 * front.max(axis=0)).astype(np.float32)                 # [1.1, 1.1]
hv_exe = evoxc.build(
    hv,
    TensorSpec((), np.int64),                                       # leading RNG key, scalar int64
    spec_objs,
    TensorSpec(ref.shape, np.float32),
    200000,                                                         # num_sample (static arg)
    backend="numpy",
)
print(f"ref: {ref}")
print(f"HV : {float(evoxc.run(hv_exe, key(0), objs, ref, 200000).numpy()):.6f}")
```

```text
true front: (2000, 2)
IGD: 0.004283
GD : 0.000186
ref: [1.1 1.1]
HV : 0.417172
```

- **IGD** averages the distance from each *true-front* point to the nearest achieved point, so it
  penalizes gaps in coverage; **GD** averages the distance from each achieved point to the nearest
  front point, so it penalizes points that sit far from the optimum but ignores coverage. A small GD
  with a much larger IGD means "converged, but clustered".
- **Hypervolume** measures the volume of objective space dominated by the front, up to the reference
  point. It is the only one of the three that needs a reference point and it is **unnormalized**: the
  `ref` used here, `1.1 ×` the front's worst corner (`[1.1, 1.1]`), is the conventional safe choice.
  A reference point the front does not dominate clips every extent to zero and yields `0.0`.
- `hv` takes a **leading RNG key** (`hv(key, objs, ref, num_sample=100000)`) because it is a
  Monte-Carlo estimator; fix the key and the sample budget when comparing runs. `gd`, `gd_plus`,
  `igd` and `igd_plus` are deterministic and take no key.

```{note}
`dtlz_pf` — `evox_etl.problems.numerical.dtlz.pf` — is a graph function of the problem config only,
so `evoxc.build(pf, problem)` followed by `evoxc.run(exe, problem)` produces the analytic front. It
returns 2000 points for a two-objective DTLZ2 and 2926 for three objectives at the default
`ref_num=1000`; the achieved front has a few hundred points, so IGD is dominated by the true front's
resolution, not by the optimizer.
```

The metric reference material — argument shapes, the `*_plus` dominance-respecting variants, the two
Monte-Carlo hypervolume estimators and how to sweep `ref` — is in {doc}`../guide/advanced/metrics`.

## Plotting the front

```python
figure = workflow.monitor.plot(problem_pf=front)   # 2-D scatter of the front, true front overlaid
```

The builder is chosen by the number of objectives in the recorded fitness, and `problem_pf` is
overlaid as a reference trace:
`plot_obj_space_2d` for two objectives, `plot_obj_space_3d` for three, `plot_obj_space_1d` for one.
The same program with a three-objective DTLZ2 plots as a 3-D figure:

```python
problem_3obj = DTLZ2(d=12, m=3)
workflow_3obj = StdWorkflow(
    make_nsga2(pop_size=100, n_objs=3, lb=np.zeros(12), ub=np.ones(12)), problem_3obj,
    monitor=EvalMonitorConfig(),
    opt_direction=["min", "min", "min"],
    num_generations=50,
    backend="numpy",
)
workflow_3obj.run(seed=42)
front_3obj = evoxc.run(evoxc.build(dtlz_pf, problem_3obj), problem_3obj).numpy()  # (2926, 3)
figure = workflow_3obj.monitor.plot(problem_pf=front_3obj)                        # 3-D figure
```

```{tip}
`plot()` needs the fitness history only — `full_sol_history=True` is required for `get_pf()` but not
for the figure. The other figure builders (`plot_dec_space`, the animated variants) and the
animation options are covered in {doc}`../guide/advanced/viz`.
```

## Exporting a run to `.exv`

EvoXVision (`.exv`) is a streaming binary format holding each generation's population and fitness,
so a run can be replayed or inspected outside Python.
`new_exv_metadata` derives the schema from the first two generations, then `EvoXVisionAdapter`
streams one chunk per generation:

```python
from evox_etl.vis_tools import EvoXVisionAdapter, new_exv_metadata

populations = [np.asarray(p) for p in workflow.monitor.get_solution_history()]
fitnesses = [np.asarray(f) for f in workflow.monitor.get_fitness_history()]
print(f"{len(populations)} generations of {populations[0].shape} populations")

adapter = EvoXVisionAdapter("dtlz2_run.exv")
adapter.set_metadata(
    new_exv_metadata(populations[0], populations[1], fitnesses[0], fitnesses[1])
)
adapter.write_header()
for population, generation_fitness in zip(populations, fitnesses):
    adapter.write(population.tobytes(), generation_fitness.tobytes())
adapter.flush()
adapter.writer.close()
```

```text
50 generations of (100, 12) populations
```

The file starts with the magic number `exv1`; for 50 generations of 100 × 12 `float32` populations
and 100 × 2 fitness values it is about 280 kB (fixed size per chunk — there is no compression).
Both histories require `full_sol_history=True` (solutions) and the default `full_fit_history=True`
(fitness); the workflow records them host-side after each compiled generation.

The format, its header layout and the read-back path are documented in
{doc}`../guide/advanced/viz`.

## Choosing the algorithm

NSGA-II is only one of the six built-in multi-objective algorithms; each has a `make_*` constructor
taking the same `(pop_size, n_objs, lb, ub)` core arguments:

| Constructor | Algorithm | Note |
|---|---|---|
| `make_nsga2` | NSGA-II | Fast non-dominated sort + crowding distance; the default choice |
| `make_nsga3` | NSGA-III | Reference-point based, scales better to many objectives |
| `make_moead` | MOEA/D | Decomposition-based; it generates its own Das-Dennis weight vectors from `pop_size`/`n_objs` (and the effective population becomes that count) |
| `make_rvea` / `make_rveaa` | RVEA / RVEA-a | Reference-vector guided, adaptive |
| `make_hype` | HypE | Monte-Carlo hypervolume-based selection |

The `opt_direction` list must have one entry per objective and `n_objs` must match it, or the
monitor and the algorithm disagree about the fitness shape.

## Where to go next

- Metrics in depth, including `gd_plus` / `igd_plus`: {doc}`../guide/advanced/metrics`.
- Visualization and the `.exv` read/write API: {doc}`../guide/advanced/viz`.
- Tuning a workflow's hyperparameters host-side, with `HPOProblemWrapper` + `random_search`:
  {doc}`../guide/advanced/hpo`.
- Single-objective walkthrough: {doc}`so`.
