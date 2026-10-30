# Examples

Runnable, end-to-end examples for the **EvoX 2.0** functional API.
Each page is a short program you can paste into a file, run, and then modify; every
printed number on these pages was produced by actually running the snippet with
`seed=42` on the NumPy backend.

```{warning} Experimental EvoX 2.0 pre-release
These examples target the **EvoX 2.0 pre-release** — the functional rewrite of EvoX on top of the
**EvoX Compiler** (`evoxc`) tensor library, importable as **`evox`**.
They run on the **NumPy backend by default** (`backend="numpy"`), so no compiler toolchain or GPU
is needed to follow along.
The 1.x object-oriented torch API (`ModuleBase`, `Parameter`/`Mutable`,
`torch.compile`/`vmap`, the Brax/MuJoCo/supervised-learning problems) was removed in 2.0 and is not
used anywhere in these pages.
APIs may still change between pre-releases — see {doc}`../tutorial/tutorial_part7` for the
migration view.
```

## Before you start

The examples assume the repository layout, so add `src/` to the import path instead of installing
EvoX:

```bash
PYTHONPATH=src python my_example.py
```

Two optional dependencies are worth installing alongside EvoX:

| Package | Needed for | Without it |
|---|---|---|
| `plotly` | `wf.monitor.plot()` and the figure builders in `evox.vis_tools` | `plot()` warns and returns `None`; the `.exv` serializer and everything else still work |
| `evoxc` compilers (IREE / TVM / XLA) | a non-NumPy `backend=` | the default NumPy backend still runs the same program |

```{note}
Every algorithm, problem and monitor in 2.0 is a **frozen config dataclass** built by a `make_*`
constructor; there are no component classes to subclass and no `Parameter`/`Mutable` bookkeeping.
A run is therefore always the same four objects: a config, a problem config, a monitor config and
one `StdWorkflow`.
```

## The examples

```{toctree}
:maxdepth: 1
so
mo
neuroevolution
```

| Page | What it shows |
|---|---|
| {doc}`so` | Single-objective optimization: PSO on Sphere step by step, a measured PSO / DE / CMA-ES comparison, convergence plotting, and swapping in `Ackley` / `Rastrigin`. |
| {doc}`mo` | Multi-objective optimization: NSGA-II on DTLZ2, reading the Pareto front, scoring it with IGD/GD/hypervolume, plotting it, and exporting a run to an `.exv` file. |
| {doc}`neuroevolution` | The virtual (training-free) ES family: `VirtualES` / `VirtualLoRAES` training a small network through a `VirtualProblem`, and the Gaussian-noise payload protocol that makes it work without materializing a population. |

These pages deliberately stay shallow in places: the deep dive into how a run is put together is in
{doc}`../tutorial/tutorial_part2`, backend and device detail (compiled extras, PJRT plugin,
performance) is in {doc}`../guide/advanced/backends`, and the reference material for the analysis
pieces lives in {doc}`../guide/advanced/metrics`, {doc}`../guide/advanced/viz` and
{doc}`../guide/advanced/hpo`.
Writing your own algorithm or problem from scratch is covered in
{doc}`../guide/developer/custom-alg-pro`.
