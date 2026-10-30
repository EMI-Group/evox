# Welcome to EvoX's documentation!

```{toctree}
:caption: 'Getting Started'
:maxdepth: 1
:hidden:

tutorial/index
guide/install/index
guide/developer/index
guide/advanced/index
```

```{toctree}
:caption: 'Additional Resources'
:maxdepth: 2
:hidden:

apidocs/index
Examples <examples/index>
Miscellaneous <miscellaneous/index>
```

[[English Version]](https://evox.readthedocs.io/en/latest/)   [[中文版本]](https://evox.readthedocs.io/zh-cn/latest/)

```{warning} EvoX 2.0 — experimental pre-release
These pages document the **EvoX 2.0 pre-release**, a functional rewrite of EvoX on top of the
**EvoX Compiler** (`evoxc`) tensor library. The import name is **`evox_etl`**; the 1.x object-oriented
torch API (`import evox`, `ModuleBase`, `Parameter`/`Mutable`, `torch.compile`/`vmap`) is **no longer
documented and will be removed**. APIs may still change between pre-releases — see
{doc}`tutorial/tutorial_part7` if you are bringing 1.x code across.
```

EvoX is a GPU-accelerated framework for scalable evolutionary computation.

The 2.0 programming model is **functional end to end**:

- **Configuration is data**: every algorithm, problem and monitor is a frozen config dataclass,
  usually built by a `make_*` constructor (e.g. `make_pso(...)`, `make_cma_es(...)`).
- **State is data**: `init(config, key)` returns a tensor state; `step(config, state, evaluate)`
  returns the next state. Nothing is stored on the object.
- **One compile site**: `StdWorkflow` traces the algorithm/problem/monitor functions once through
  `evoxc.build` and runs the resulting program in a loop; monitors and metrics stay host-side.
- **Explicit randomness**: stateless RNG keys (`evox_etl.key(seed)`) live in the state — there is no
  global seed.

## Quickstart

```python
from evox_etl import StdWorkflow, EvalMonitorConfig
from evox_etl.algorithms import make_pso
from evox_etl.problems import Sphere

dim = 10
workflow = StdWorkflow(
    algorithm=make_pso(100, lb=[-100.0] * dim, ub=[100.0] * dim),
    problem=Sphere(),
    monitor=EvalMonitorConfig(),
    opt_direction="min",
    num_generations=50,
    backend="numpy",          # default: no compiled toolchain required
)
state = workflow.run(50, seed=42)
print(workflow.monitor.get_best_fitness())
workflow.monitor.plot()       # Plotly figure
```

See {doc}`tutorial/tutorial_part2` for the walkthrough and {doc}`tutorial/tutorial_part6` for running
the same program on IREE, TVM or XLA devices.

## Key Features

- 🚀 **Compiled, not interpreted**: the same user program runs on the default NumPy interpreter or is
  compiled through IREE / TVM / XLA for CPU and GPU execution — only the `backend=` argument changes.
- 🌐 **Versatile optimization suite**:
  - Single- and multi-objective algorithms: 30+ algorithms spanning DE, ES, PSO, the training-free
    virtual ES family, and NSGA-II/III, MOEA/D, RVEA, RVEA-a, HypE.
  - 30+ benchmark problems: basic numerical functions, DTLZ1–7, CEC2022, plus virtual
    (Gaussian-noise neuroevolution) problems.
  - Scale up with the broader **EvoX ecosystem** — EvoX plus 13 related GPU-accelerated EC libraries —
    providing **140+ algorithms** and **240+ benchmark problems/environments**.
- 🛠️ **Designed for simplicity and reproducibility**: no `Parameter`/`Mutable` bookkeeping, plain
  functions you can test and reuse, states you can serialize, and deterministic runs from a single
  integer seed.
- 📊 **Built-in analysis**: `EvalMonitor` records fitness/solution histories and auxiliary channels,
  `evox_etl.metrics` provides GD/IGD/HV, and `evox_etl.vis_tools` plots and serializes results to
  `.exv` files.

---

```{eval-rst}
.. grid:: 3
   :gutter: 1 2 3 5
   :padding: 1

   .. grid-item-card:: :octicon:`people;1.5em` Tutorials
      :padding: 3
      :margin: 0
      :link: tutorial/index
      :link-type: doc

   .. grid-item-card:: :octicon:`desktop-download;1.5em` Installation Guide
      :padding: 3
      :margin: 0
      :link: guide/install/index
      :link-type: doc

   .. grid-item-card:: :octicon:`mortar-board;1.5em` Developer Guide
      :padding: 3
      :margin: 0
      :link: guide/developer/index
      :link-type: doc

.. grid:: 3
   :gutter: 1 2 3 5
   :padding: 1

   .. grid-item-card:: :octicon:`cpu;1.5em` Backends, Metrics, HPO
      :padding: 3
      :margin: 0
      :link: guide/advanced/index
      :link-type: doc

   .. grid-item-card:: :octicon:`list-unordered;1.5em` API Reference
      :padding: 3
      :margin: 0
      :link: apidocs/index
      :link-type: doc

   .. grid-item-card:: :octicon:`beaker;1.5em` Examples
      :padding: 3
      :margin: 0
      :link: examples/index
      :link-type: doc
```
