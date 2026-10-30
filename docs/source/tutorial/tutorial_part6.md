# 6. Multi-Objective Optimization

In {doc}`tutorial_part3` every candidate was scored by a single number, so "better" was a total order and the monitor could keep one best value.
Real problems rarely work like that: a design is judged by cost, mass and stiffness at the same time, and those pull against each other.
In a **multi-objective (MO)** problem a solution evaluates to a *vector*, no solution wins on every axis, and the answer is a **Pareto front** — the set of solutions for which no objective can be improved without making another one worse.

This chapter runs a complete MO study: NSGA-II on the DTLZ2 benchmark, reading the resulting front, scoring it with quality indicators, plotting it, and saving it for the EvoXVision viewer.
It targets the experimental **EvoX 2.0 pre-release**, the functional rewrite of EvoX on the `evoxc` compiler (`import evox_etl`).

```{warning} Pre-release API
Everything below uses the EvoX 2.0 functional API (`import evox_etl`, frozen config dataclasses, `make_*` constructors, explicit RNG keys).
The 1.x object-oriented torch API (`import evox`, `ModuleBase`, `Parameter`/`Mutable`) was removed in 2.0 and is not used anywhere in this chapter — see {doc}`tutorial_part7` if you are bringing 1.x code across.
```

## 6.1 What multi-objective mode changes

Only four things change; the workflow, monitor and plotting calls are the ones you already know.

| Lever | Single-objective | Multi-objective |
|---|---|---|
| `opt_direction` | `"min"` (a string) | `["min", "min", "min"]` — **a list**, one entry per objective. The list is also what flips the monitor into MO mode: the workflow fills in `multi_obj=True` and `n_obj=len(opt_direction)`. |
| Algorithm config | `make_pso(pop_size, lb, ub)` | `make_nsga2(pop_size, n_objs, lb, ub)` — the algorithm must know how many objectives it is ranking. |
| Monitor | keeps best-so-far | keeps per-generation fitness (and solutions, with `full_sol_history=True`), and exposes Pareto-front accessors instead of a single best. |
| What you read | `get_best_fitness()` | `get_pf()` plus the quality indicators `gd` / `igd` / `hv`. |

`evox_etl` still minimizes internally: the workflow applies `opt_direction` to the fitness before it reaches the algorithm, and the monitor un-applies it on the way out.
Mixed directions are allowed, e.g. `opt_direction=["min", "max"]`, and each entry is sign-flipped independently.

```{warning} Always pass a list for MO
`opt_direction="min"` on a three-objective problem does **not** raise — the optimizer runs, but the monitor stays in single-objective mode (`multi_obj=False`, `n_obj=None`).
Every MO accessor then fails, e.g. `get_pf()` raises `ValueError: get_pf is only available for multi-objective optimization.` and `get_best_fitness()` fails with an `AttributeError`.
```

## 6.2 A complete run: NSGA-II on DTLZ2

DTLZ2 is a standard three-objective benchmark: with `d` variables and `m` objectives the last `d - m + 1` variables only drive a distance function, and the optimal front is the section of the unit sphere in the positive octant.

```python
import time

import numpy as np

from evox_etl import EvalMonitorConfig, StdWorkflow
from evox_etl.algorithms import make_nsga2
from evox_etl.problems.numerical import DTLZ2

DIM, N_OBJ, POP_SIZE, GENERATIONS = 12, 3, 100, 100

problem = DTLZ2(d=DIM, m=N_OBJ, ref_num=200)

workflow = StdWorkflow(
    make_nsga2(pop_size=POP_SIZE, n_objs=N_OBJ, lb=[0.0] * DIM, ub=[1.0] * DIM),
    problem,
    monitor=EvalMonitorConfig(full_sol_history=True),
    opt_direction=["min"] * N_OBJ,
    num_generations=GENERATIONS,
)


def report(generation: int) -> None:
    latest = workflow.monitor.get_latest_fitness()  # (pop_size, n_objs), un-negated
    print(f"gen {generation:3d}  mean objectives {np.round(latest.mean(axis=0), 3)}"
          f"  ({time.time() - t0:.1f} s)")


t0 = time.time()
state = workflow.init(seed=42)
state = workflow.init_step(state)
report(1)
for generation in range(2, GENERATIONS + 1):
    state = workflow.step(state)  # one full NSGA-II generation
    if generation % 20 == 0:
        report(generation)
```

Output (one run on a laptop CPU; the wall-clock times vary by machine):

```text
gen   1  mean objectives [0.759 0.694 1.226]  (0.2 s)
gen  20  mean objectives [0.519 0.525 0.513]  (0.3 s)
gen  40  mean objectives [0.534 0.472 0.549]  (0.4 s)
gen  60  mean objectives [0.47  0.487 0.513]  (0.5 s)
gen  80  mean objectives [0.547 0.529 0.408]  (0.6 s)
gen 100  mean objectives [0.483 0.504 0.564]  (0.7 s)
```

The means start near the centre of the box the initial population is drawn from and settle around `0.5` per objective, which is the centroid of a uniformly covered DTLZ2 front (for points uniform on the sphere, `E|x_i| = 1/2`): the population is spread over the front instead of collapsing into one corner.
Section 6.4 replaces this crude progress view with real indicators.

| Argument | Value here | Why |
|---|---|---|
| `pop_size=100` | 100 | NSGA-II ranks and crowds the whole population every generation; 100 is the classic size that still covers a three-objective front reasonably. |
| `n_objs=3` | 3 | Must equal the number of objectives the problem returns (`m=3`); it sizes the algorithm's fitness/rank/crowding buffers. |
| `lb` / `ub` | `0.0` / `1.0`, length 12 | DTLZ2's decision variables live in `[0, 1]^d`; `len(lb)` is the dimension the initial population is drawn in. |
| `DTLZ2(d=12, m=3, ref_num=200)` | `d=12` | 3 objectives + 10 distance variables; `ref_num` only sizes the truth front used for the metrics later (595 points). |
| `opt_direction=["min"] * 3` | list | Switching the monitor to MO mode; all DTLZ objectives are minimized. |
| `EvalMonitorConfig(full_sol_history=True)` | on | `get_pf()` needs the solutions, not only the fitness: with the default (`False`) it warns and then fails with `ValueError: need at least one array to concatenate`. Only `get_pf_fitness()` works from the fitness history alone. |
| `backend` (not passed → `"numpy"`) | numpy | The default interpreter backend needs no compiled toolchain, so the example runs anywhere. The identical program runs unchanged on the compiled `iree` / `xla` backends once their packages are installed (see {doc}`tutorial_part4`). |

`workflow.run(seed=42)` performs the same 100 generations (the last one through `final_step` when the algorithm module defines it — the MO algorithms do not).

## 6.3 Reading the Pareto front

The monitor accumulated 100 generations of fitness and solutions. `get_pf()` merges them and keeps only the non-dominated solutions:

```python
sol, fit = workflow.monitor.get_pf()
print(sol.shape, sol.dtype, "|", fit.shape, fit.dtype)
print("fitness range per objective:",
      np.round(fit.min(axis=0), 4), "→", np.round(fit.max(axis=0), 4))
```

```text
(2301, 12) float32 | (2301, 3) float32
fitness range per objective: [0. 0. 0.] → [1.0022 1.0026 1.0025]
```

The front spans every objective from `0` (each objective can be driven to zero by sacrificing the other two) up to `1.0025`; a population that had collapsed onto a single point would show a much narrower range.
`sol` rows are aligned with `fit` rows: row `i` is the decision vector that produced objective vector `i`.

| Accessor | Returns | Notes |
|---|---|---|
| `get_pf(deduplicate=True)` | `(solutions, fitness)` | Deduplicates on **solutions**, then keeps non-domination rank 0. `(2301, 12)` / `(2301, 3)` here. |
| `get_pf_fitness(deduplicate=True)` | `(k, n_objs)` | Deduplicates on **fitness** instead, then ranks; works from the fitness history alone. `(2301, 3)` here. |
| `get_pf_solutions(deduplicate=True)` | `(k, dim)` | Convenience wrapper around `get_pf`. `(2301, 12)` here. |
| `get_fitness_history()` | list of `(pop_size, n_objs)` | One array per generation, un-negated. `100 × (100, 3)` here. |
| `get_solution_history()` | list of `(pop_size, dim)` | One array per generation; requires `full_sol_history=True`. |
| `get_latest_fitness()` / `get_latest_solution()` | `(pop_size, n_objs)` / `(pop_size, dim)` | The last generation only. |

The single-objective accessors (`get_best_fitness`, `get_best_solution`, `get_topk_fitness`, `get_topk_solutions`) raise `ValueError: Multi-objective optimization does not have a single best solution. Please use get_pf_solutions` — there is no "best" point on a front.

```{note} The front is ranked over *everything ever evaluated*
The buffers hold `pop_size × generations = 10 000` fitness rows here, of which `9 950` are distinct; the host-side non-dominated sort then leaves `2 301`.
That ranking costs about 36 s on the numpy backend in this example — far more than the 0.8 s the 100 generations themselves take.
It is an end-of-run analysis tool: call `get_pf()` once and reuse `sol` / `fit`, and leave `full_sol_history=False` (or keep runs short) while you are still iterating on a script. `get_pf_fitness()` and `get_pf_solutions()` each redo the ranking.
```

## 6.4 Quality indicators

Visual inspection does not scale, so MO runs are scored with indicators from `evox_etl.metrics`.
They are traced `evoxc` functions like everything else in the framework: they take symbolic tensors while the graph is traced and concrete tensors when it runs, so the shortest way to call one is `evoxc.evaluate(fn, *concrete_args)`, which builds and runs a one-shot program and returns a scalar `Tensor` (use `.numpy()`).

| Function | Signature | Meaning |
|---|---|---|
| `gd` | `gd(objs, pf)` | Generational Distance: distance from each approximation point to the closest true-front point (L2 norm of the per-solution distances, divided by `n`). **Convergence**. Lower is better. |
| `gd_plus` | `gd_plus(objs, pf, p=1.0)` | GD+, the dominance-respecting variant: each solution's distance to its closest front point sums only the components where the front point is **worse** (`max(pf - obj, 0)`), so a front point that is better contributes nothing and a solution weakly dominated by a front point scores 0. Lower is better. |
| `igd` | `igd(objs, pf, p=1.0)` | Inverted GD: distance from each true-front point to the closest approximation point. Because *every* front point must be covered, it rewards **convergence and diversity**. Lower is better. |
| `igd_plus` | `igd_plus(objs, pf, p=1.0)` | IGD+ with the same dominance-respecting distance, reversed (`max(obj - pf, 0)`). Lower is better. |
| `hv` | `hv(key, objs, ref, num_sample=100000)` | Hypervolume, estimated by bounding-cube Monte Carlo. Unlike the others it takes a **leading RNG key** and a reference point `ref`. **Higher** is better. |
| `bounding_cube_monte_carlo_hv` | `(key, objs, ref, num_sample=100000)` | The estimator `hv` delegates to; same result, explicit name. |
| `each_cube_monte_carlo_hv` | `(key, objs, ref, num_sample=100000)` | One cube per front point, with `num_sample` split across the points; an alternative estimator for small fronts. |

```python
import evoxc

import evox_etl
from evox_etl.metrics import gd, gd_plus, hv, igd, igd_plus
from evox_etl.problems.numerical.dtlz import pf as dtlz_pf

truth_front = evoxc.evaluate(lambda: dtlz_pf(problem)).numpy()  # (595, 3), the analytic front
print("truth front", truth_front.shape)

print(f"IGD   {float(evoxc.evaluate(igd, fit, truth_front).numpy()):.6f}")
print(f"IGD+  {float(evoxc.evaluate(igd_plus, fit, truth_front).numpy()):.6f}")
print(f"GD    {float(evoxc.evaluate(gd, fit, truth_front).numpy()):.6f}")
print(f"GD+   {float(evoxc.evaluate(gd_plus, fit, truth_front).numpy()):.6f}")

ref = np.array([1.1] * N_OBJ, dtype=np.float32)  # must dominate every point of the front
key = evox_etl.key(0)  # explicit RNG key: it is part of the program, so results are reproducible
hv_value = evoxc.evaluate(lambda k, o, r: hv(k, o, r, num_sample=10_000), key, fit, ref)
print(f"HV    {float(hv_value.numpy()):.6f}")
```

```text
truth front (595, 3)
IGD   0.019138
IGD+  0.012776
GD    0.000544
GD+   0.004072
HV    0.774110
```

Reading the numbers:

- **GD ≈ 5e-4**: every one of the 2 301 approximation points sits about half a thousandth away from the true front — the population has converged.
- **IGD ≈ 0.019**: the reverse direction is worse, i.e. some regions of the true front are still uncovered — `0.019` is about a third of the `0.051` average spacing between neighbouring front points, so the coverage is decent but not gap-free. GD alone would have hidden that, which is why IGD is the usual reporting choice.
- **GD+/IGD+** use the dominance-respecting distance, so they never reward a point for overshooting the front. Both are larger than their plain counterparts here because the approximation slightly overshoots (`max(fit) ≈ 1.0025 > 1`).
- **HV ≈ 0.774**: the dominated volume inside the reference cube. The reference point must be worse than every solution; the true DTLZ2 front lies on the unit sphere (every coordinate `≤ 1`) and this approximation only overshoots to `1.0025`, so `[1.1, 1.1, 1.1]` dominates all of it (cube volume `1.331`). With the same key and sample count the *true* front scores `0.780363`, so this run is within about `0.006` of the best possible value.

`num_sample` is the Monte-Carlo budget of the hypervolume estimators: `100 000` samples (the default) cost about 5 s on this front and move the estimate from `0.774110` (`num_sample=10_000`) to `0.775148`.
The other indicators are exact, so only `hv` needs the budget knob.
Because `num_sample` is a plain Python integer (not a tensor) and `evoxc.evaluate` derives one spec per concrete tensor argument, the static value is bound through the small `lambda` above.

## 6.5 Plotting the front

The monitor carries a `plot` helper that dispatches on the number of objectives: one objective gives the convergence plot of {doc}`tutorial_part3`, two give a 2-D scatter, three a 3-D scatter, and anything else warns and returns `None`.
Pass the analytic front as `problem_pf` to draw it underneath the population.

```python
fig = workflow.monitor.plot(problem_pf=truth_front)
print(type(fig).__name__, len(fig.data), "traces,", len(fig.frames), "frames")
fig.show()
```

```text
Figure 2 traces, 100 frames
```

The figure holds one frame per generation (`Pareto Front` plus the moving `Population`), so the animation slider replays the whole optimization.
`source="eval"` (the default) plots what the problem sees, i.e. un-negated fitness.

For full control, call the figure builders in `evox_etl.vis_tools` directly — they take the host-side history list:

```python
from evox_etl.vis_tools import plot_obj_space_2d, plot_obj_space_3d

history = workflow.monitor.get_fitness_history()  # 100 × (100, 3)

fig3 = plot_obj_space_3d(history, problem_pf=truth_front)
fig2 = plot_obj_space_2d(  # 2-D builder: project onto the first two objectives
    [f[:, :2] for f in history], problem_pf=truth_front[:, :2]
)
print(len(fig3.frames), len(fig2.frames), [trace.name for trace in fig2.data])
```

```text
100 100 ['Pareto Front', 'Population']
```

| Builder | Use for |
|---|---|
| `plot_obj_space_2d(fitness_history, problem_pf=None, sort_points=False)` | 2-objective fronts (what we just called). |
| `plot_obj_space_3d(fitness_history, problem_pf=None, sort_points=False)` | 3-objective fronts; `monitor.plot` uses this for `n_obj == 3`. |
| `plot_obj_space_1d(fitness_history, animation=True)` | single-objective convergence; the `_1d_no_animation` / `_1d_animation` variants fix the mode. |
| `plot_dec_space(population_history)` | the same run in decision space, to see how the population moves through the search box. |

```{note} Plotting needs the optional extra
Plotly ships as the `vis` extra (`pip install "evox[vis]"`).
Without it, `workflow.monitor.plot()` warns and returns `None`, and the `plot_obj_space_*` builders raise an `ImportError` — the run itself is unaffected.
```

## 6.6 Saving the run with EvoXVision (`.exv`)

The `.exv` format stores a whole run — every generation's population and fitness — in one binary file that the EvoXVision viewer can open and replay.
`new_exv_metadata` builds the schema from two generations (the first one may have a different population size or dtype than the rest), then `EvoXVisionAdapter` streams the chunks:

```python
import tempfile
from pathlib import Path

from evox_etl.vis_tools import EvoXVisionAdapter, new_exv_metadata

solutions = workflow.monitor.get_solution_history()  # 100 × (100, 12)
fitnesses = workflow.monitor.get_fitness_history()  # 100 × (100, 3)

metadata = new_exv_metadata(solutions[0], solutions[1], fitnesses[0], fitnesses[1])
path = Path(tempfile.mkdtemp()) / "nsga2_dtlz2.exv"

adapter = EvoXVisionAdapter(path, buffering=0)
adapter.set_metadata(metadata)
adapter.write_header()
adapter.write(solutions[0].tobytes(), fitnesses[0].tobytes())
for solution, fitness in zip(solutions[1:], fitnesses[1:]):
    adapter.write(solution.tobytes(), fitness.tobytes())
adapter.flush()
adapter.writer.close()

print(path.name, path.stat().st_size, "bytes;",
      metadata["n_objs"], "objectives;",
      metadata["initial_iteration"]["chunk_size"], "bytes per generation")
```

```text
nsga2_dtlz2.exv 600536 bytes; 3 objectives; 6000 bytes per generation
```

A generation's chunk is just the raw little-endian bytes of the population (`100 × 12 × 4 = 4 800`) followed by those of the fitness (`100 × 3 × 4 = 1 200`), which is why the schema can be computed once, up front.
The file starts with the magic `exv1`, a 32-bit little-endian metadata length and the JSON schema; `buffering=0` writes through immediately, so a long run can be inspected while it is still going.
Drag the file into the EvoXVision viewer to scrub through generations in objective or decision space.

## 6.7 Choosing a multi-objective algorithm

All MO algorithms share the same config-plus-`make_*` shape, so switching is a one-line edit.

| Constructor | Good for |
|---|---|
| `make_nsga2(pop_size, n_objs, lb, ub)` | 2–3 objectives; non-dominated rank plus crowding distance — the sane default, and what this chapter used. |
| `make_nsga3(pop_size, n_objs, lb, ub)` | many objectives (4+): reference points replace crowding distance so the front stays well spread instead of drifting to the middle. |
| `make_rvea(pop_size, n_objs, lb, ub, alpha=2.0, fr=0.1, max_gen=100)` | many objectives; reference-vector guided with an angle-penalised distance (`alpha`), and `fr` sets how often the reference vectors adapt. |
| `make_rveaa(pop_size, n_objs, lb, ub, alpha=2.0, fr=0.1, max_gen=100)` | RVEA plus the reference-vector regeneration strategy, for many-objective problems with an irregular front. |
| `make_moead(pop_size, n_objs, lb, ub)` | decomposition into scalar subproblems; very fast, front resolution bounded by the population. |
| `make_hype(pop_size, n_objs, lb, ub, n_sample=10000)` | hypervolume-driven search for many objectives, with `n_sample` as the Monte-Carlo budget per generation. |

```{tip} Swapping is one line
Replace `make_nsga2(...)` with `make_nsga3(...)` (or `make_rvea`, `make_moead`, `make_hype`) and the workflow, monitor, metrics, plotting and `.exv` export above all keep working — the config field names and the step protocol are the same.
The same holds for the problem: `DTLZ2(d=12, m=3, ref_num=200)` becomes `DTLZ4(...)`, `DTLZ7(...)` or any of the other benchmark configs.
```

Two honest caveats: RVEA and RVEAa size their population by the Das–Dennis reference-vector count, so the effective population can be slightly smaller than `pop_size` (90 instead of 100 for three objectives), and HypE's `n_sample` budget directly multiplies its per-generation cost.

## 6.8 Next steps

You now have the full MO loop: configure, run, read the front, score it, plot it, export it.

- {doc}`../guide/advanced/metrics` — the indicators in depth: `p` exponents, reference-point selection, and how the Monte-Carlo hypervolume estimators behave.
- {doc}`../guide/advanced/viz` — every `vis_tools` figure builder and the `.exv` format in detail.
- {doc}`../guide/advanced/hpo` — hyperparameter optimization, the other place where you will meet a list-valued objective.
- {doc}`tutorial_part4` — the algorithm catalogue, the `make_*` constructors and custom operator injection.
- {doc}`tutorial_part5` — the problem catalogue, including how to write your own problem.
- {doc}`../guide/advanced/backends` — the compiled IREE / TVM / XLA backends and their options.
- {doc}`../guide/developer/custom-alg-pro` — writing your own algorithm or problem.
- {doc}`../guide/developer/custom-monitor` — writing your own monitor.
