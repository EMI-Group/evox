# EvoX Tutorial

```{warning} EvoX 2.0 — experimental pre-release
These pages document the **EvoX 2.0 pre-release**: a functional rewrite of EvoX on top of the
**EvoX Compiler** (`evoxc`) tensor library. The public import name is **`evox_etl`**.
The 1.x object-oriented torch API (`import evox`, `ModuleBase`, `Parameter`/`Mutable`,
`torch.compile`/`vmap`, class-based `StdWorkflow` components) has been removed and is no longer
taught anywhere in this tutorial. Because 2.0 is still a pre-release, APIs may change between
pre-releases — pin a version if you need stability.
```

This tutorial takes you from a clean Python environment to a running, compiled evolutionary
optimization loop. Every page is based on the same programming model: **configuration is data**
(frozen dataclasses), **state is data** (a pytree of tensors), and **the only compile site is
`StdWorkflow`**, which traces the algorithm, problem and monitor once through `evoxc` and then
runs the resulting program in a host-side loop.

Read the parts in order the first time. Parts 1–2 get you installed and through one complete
program; after that you can jump to whichever part matches what you are doing.

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Part
     - Who it is for
   * - :doc:`tutorial_part1`
     - Everyone. What EvoX 2.0 is, what changed from 1.x, how to install it, and a runnable smoke test.
   * - :doc:`tutorial_part2`
     - Everyone. The full walkthrough of one program: algorithm, problem, monitor and workflow.
   * - :doc:`tutorial_part3`
     - Users writing their first real optimization script.
   * - :doc:`tutorial_part4`
     - Users configuring algorithms, problems and monitoring for their own tasks.
   * - :doc:`tutorial_part5`
     - Users extending the framework with their own algorithms, problems, operators and monitors.
   * - :doc:`tutorial_part6`
     - Anyone who needs speed: running the same program on the IREE, TVM or XLA backends and on GPUs.
   * - :doc:`tutorial_part7`
     - Users bringing 1.x torch code across, plus practical end-to-end examples.
```

```{toctree}
:maxdepth: 1
tutorial_part1
tutorial_part2
tutorial_part3
tutorial_part4
tutorial_part5
tutorial_part6
tutorial_part7
```

## What you will be using

- **`evox_etl`** — the framework itself: `StdWorkflow`, `EvalMonitor`/`EvalMonitorConfig`,
  the `core` protocols, `metrics` (GD/IGD/HV), `utils`, `vis_tools`, `workflows`, and the
  `algorithms` / `problems` / `operators` packages loaded through extension autoloading.
- **`evoxc`** — the EvoX Compiler. It traces your program into a graph and executes it on a
  backend of your choice (`backend=` at the workflow): the default self-contained NumPy
  interpreter, or a compiled IREE / TVM / XLA backend for CPUs and GPUs.
- **`evox_etl_ext`** — the plugin namespace package. Installed extensions are merged into the
  matching `evox_etl.*` modules when you `import evox_etl`; see {doc}`tutorial_part5`.

EvoX ships **30+ algorithms** and **30+ benchmark problems**. Together with the 13 related
GPU-accelerated evolutionary computation libraries in the EMI-Group ecosystem, the combined
collection offers **140+ algorithms** and **240+ benchmark problems/environments**.

```{tip}
If you only read one page, read {doc}`tutorial_part2`: it is the complete program, annotated.
```
