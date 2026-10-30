# 1. EvoX 2.0 and Installation

## What EvoX is — and what changed in 2.0

EvoX is a GPU-accelerated framework for scalable evolutionary computation. In 1.x every component
was a `ModuleBase` subclass holding `Parameter`/`Mutable` tensors, and parallelism came from
`torch.compile`/`vmap` wrappers around those classes; in 2.0 that component model is gone.
An algorithm, a problem and a monitor are now **frozen config dataclasses** (plain hyperparameters,
built by `make_*` constructors such as `make_pso`), the population and any other mutable quantities
are **tensor-valued state** carried through plain `init` / `step` functions, and `StdWorkflow`
compiles that trio exactly once through the **`evoxc`** compiler before running it in a loop.
The package you import is **`evox`** — the torch object-oriented API of 1.x has been removed.

## The functional model at a glance

- **Algorithm = config dataclass.** `PSO(pop_size=40, lb=..., ub=...)` is inert data: no state, no
  `forward`, nothing to reset. The search dimension is implied by the length of `lb`/`ub`.
- **Problem = config dataclass.** `Sphere()` carries the problem definition; the evaluation logic
  lives in the module's `evaluate` function.
- **State = a pytree of tensors.** `init(config, key)` returns it, `step(config, state, evaluate)`
  returns the next one. State leaves are `evoxc` tensors, never Python scalars, so the whole
  pytree can cross the compilation boundary.
- **Logic = plain module-level functions.** `init(config, key)`, `step(config, state, evaluate)`,
  optional `init_step` / `final_step`, plus `evaluate` for problems. They are ordinary Python
  functions — no decorators to remember, and trivially unit-testable.
- **One compile site in the component model.** `StdWorkflow` traces the algorithm, problem and
  monitor into a graph once (`evoxc.build`) and reuses that executable for every generation;
  monitors, metrics and history bookkeeping stay host-side. (Pages that call `evoxc.build`
  themselves, such as the Pareto-front and metric examples, compile their own small graphs.)

```python
from evox import StdWorkflow, EvalMonitorConfig
from evox.algorithms import PSO
from evox.problems.numerical import Sphere

algorithm = PSO(pop_size=40, lb=[-5.0] * 5, ub=[5.0] * 5)   # frozen config dataclass
problem = Sphere()                                          # frozen config dataclass
workflow = StdWorkflow(algorithm, problem, monitor=EvalMonitorConfig(), backend="numpy")
workflow.run(generations=20, seed=42)                       # one compile, then a host-side loop
print(workflow.monitor.get_best_fitness())
```

That is the whole 2.0 programming model in eight lines. {doc}`tutorial_part2` walks through it
argument by argument.

## Installation

### Core install

```bash
pip install --pre evox==2.0.0a1
```

The distribution is still named `evox`, but 2.0 ships as **pre-releases**, so a plain
`pip install evox` resolves the 1.x stable line, which has no `evox`. `--pre` lets pip consider
pre-releases and `==2.0.0a1` names this one; `pip install "evox>=2.0.0a1"` is equivalent, because a
specifier that mentions a pre-release opts into pre-releases on its own. Either command installs the
framework together with the **NumPy backend** — `evoxc` and `numpy` are the declared dependencies — which
is the default and needs **no compiled toolchain**: there is nothing to build, no CUDA setup and no
compiler environment variable to configure. EvoX 2.0 requires **Python 3.11 or newer**, the floor
declared by both `evoxc` and the 2.0 `evox` metadata. A 3.11 environment is already enough to run every
single- and multi-objective example in this tutorial. If the pre-release is not on the index yet,
install from a checkout with `pip install -e .` instead — {doc}`../guide/install/evox` covers both
routes.

### Optional extras

```bash
pip install "evox[vis]"        # Plotly-based plotting: monitor.plot() and evox.vis_tools
pip install "evox[iree]"       # the IREE backend
pip install "evox[tvm]"        # the TVM backend
pip install "evox[compilers]"  # every compiler backend at once
```

`evox[vis]` provides Plotly (plus pandas) for the figure builders in `evox.vis_tools`, which is
what `EvalMonitor.plot()` dispatches to; without it `plot()` warns and returns `None` while all
history accessors keep working. The three compiler extras pull in the runtime that the corresponding
`evoxc` backend needs to lower and execute a graph. If you manage the compiler as its own
distribution rather than through `evox`, the same extras exist there as `evoxc[iree]`,
`evoxc[tvm]` and `evoxc[compilers]`.

### Choosing a backend

A backend is **not** a global setting. It is the `backend=` argument of the workflow that owns the
compiled step graph:

```python
workflow = StdWorkflow(algorithm, problem, monitor=EvalMonitorConfig(), backend="numpy")  # default
workflow = StdWorkflow(algorithm, problem, monitor=EvalMonitorConfig(), backend="iree")   # compiled
```

Available names are `"numpy"` (the default interpreter, always present), `"iree"`, `"tvm"` and
`"xla"`. Which devices each one supports differs; anything unsupported raises a loud
`evoxc` backend error instead of silently falling back. Pass `device=` (for example `"cpu"`,
`"cuda"` or `"cuda:0"`) to pin the step graph to a device. Per-backend capabilities and caveats are
covered in {doc}`../guide/advanced/backends`.

```{note} XLA needs a PJRT plugin
The `"xla"` backend is only usable once a PJRT plugin is available for your platform. Point the
`EVOXC_PJRT_PLUGIN` environment variable at the plugin shared library **before the process starts**:
on a cold process the backend discovery only consults that variable and a few well-known library
paths, so the per-workflow `compile_options={"plugin_path": ...}` option helps only once a PJRT
client already exists. With no plugin anywhere, building the step graph fails with a backend error
that names the missing plugin.
```

## Smoke test

Save the following as `smoke.py` and run it with `python smoke.py`. It runs 20 generations of PSO on
the sphere function, whose minimum is the origin with fitness `0`, and prints the best fitness found
so far.

```python
from evox import StdWorkflow, EvalMonitorConfig
from evox.algorithms import PSO
from evox.problems.numerical import Sphere

workflow = StdWorkflow(
    algorithm=PSO(pop_size=40, lb=[-5.0] * 5, ub=[5.0] * 5),
    problem=Sphere(),
    monitor=EvalMonitorConfig(),
    num_generations=20,
    backend="numpy",
)
workflow.run(seed=42)
print(workflow.monitor.get_best_fitness())
```

The output is a single positive float, small and well below `1.0` after 20 generations (for
example `0.009578287601470947`) — PSO is heading towards the origin. `workflow.monitor` also
exposes `get_best_solution()`, `get_fitness_history()` and, with `evox[vis]` installed, `plot()`.

If you see that number, your install works and you are ready for the full walkthrough.

```{tip} Reproducibility
`run(seed=42)` derives every random draw from that one integer through the stateless `evoxc`
key chain stored in the state. The same seed, configuration and backend give the same result —
there is no hidden global RNG to reset.
```

## Where to go next

- {doc}`tutorial_part2` — the same program, explained piece by piece.
- {doc}`../guide/install/index` — all installation options in one place.
- {doc}`../guide/install/evox` — installing EvoX itself, packaging and environments.
- {doc}`../guide/advanced/backends` — how the NumPy, IREE, TVM and XLA backends differ.
