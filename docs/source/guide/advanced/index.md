# Advanced Guide

```{toctree}
:maxdepth: 1
backends
operators
metrics
viz
hpo
extensions
```

This section goes past the guided tour. It documents the machinery around the optimizer loop: where the
compiled graph actually executes, the operator catalog you compose algorithms from, the multi-objective
quality indicators, visualization and result files, host-side hyperparameter optimization, and the plugin
system. Read {doc}`../../tutorial/tutorial_part2` first if you have not written a `StdWorkflow` yet; every
page here assumes the functional model — frozen config dataclasses, tensor-valued state, keyed randomness.

```{warning} Experimental EvoX 2.0 pre-release
These pages target the **EvoX 2.0 pre-release** (import name `evox`, compiled by `evoxc`).
They do **not** cover the EvoX 1.x torch API (`ModuleBase`, `Parameter`/`Mutable`,
`torch.compile`/`vmap`), which was removed in 2.0 and is documented nowhere in these pages.
APIs may still change between pre-releases.
```

## What each page covers

:::{list-table}
:header-rows: 1
:widths: 22 78

* - Page
  - Content
* - {doc}`backends`
  - The default NumPy interpreter versus the compiled IREE, TVM and XLA backends; pip extras and the XLA
    PJRT plugin; how `backend=`, `device=` and `compile_options=` reach `evoxc.build`; a workload-to-backend
    guidance table and the limitations that actually bite (IREE and multi-objective programs, IREE CUDA opt
    levels, large-scale CMA-ES compile time, exporter-deferred ops).
* - {doc}`operators`
  - The full `sampling` / `selection` / `crossover` / `mutation` catalog with signatures, the
    pure-function contract, the `jit_fix_operator` helpers, and how to inject custom operators into a
    multi-objective `make_*` constructor.
* - {doc}`metrics`
  - `gd`, `gd_plus`, `igd`, `igd_plus`, `hv` and the Monte-Carlo hypervolume estimators — exact signatures,
    the leading RNG key of `hv`, how to call a traced metric from host code, and a worked IGD/HV example.
* - {doc}`viz`
  - The `evox.vis_tools` Plotly figure builders, `EvalMonitor.plot()` for convergence curves and Pareto
    fronts, and the EvoXVision `.exv` binary format with a write/read round-trip.
* - {doc}`hpo`
  - Host-side hyperparameter optimization: `HPOProblemWrapper`, `HPSlot`, the HPO monitors and
    `random_search`, why it cannot live inside a traced graph, and the resulting limits.
* - {doc}`extensions`
  - The `evox_ext` PEP 420 namespace, the import-time autoloader, the functional extension contract,
    and a minimal extension package layout.
:::

## Conventions used throughout

- **Run everything on the default backend first.** All examples in this section execute with
  `backend="numpy"`, the only backend that needs nothing beyond Python and NumPy; compiled backends are
  opt-in extras, and {doc}`backends` explains when they pay off.
- **`evoxc` has no eager mode.** Tensor operations only run inside a trace (`evoxc.build` + `evoxc.run`,
  or a `StdWorkflow` step). When a page needs a tensor result on the host it shows the explicit
  build/run/`.numpy()` idiom.
- **Configuration is data.** Algorithms and problems are frozen dataclasses built by `make_*`
  constructors; their fields are static values baked into the compiled graph, not knobs you mutate later.
- **Randomness is explicit.** Every stochastic operator takes an `evoxc` PRNG key; a run is reproducible
  from the seed passed to `StdWorkflow.run(..., seed=...)`.

```{tip}
The advanced pages are independent. If you only need one thing — a metric, a plot, a backend switch — jump
straight to that page; each one is self-contained, and its standalone Python blocks have been executed
against the code in this repository. The exceptions are blocks explicitly labelled as excerpts and the
compiled-backend-only alternatives, which need an IREE/XLA install to run.
```
