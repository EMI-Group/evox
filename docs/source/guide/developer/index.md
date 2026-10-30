# Developer Guide

This section is for people who build on EvoX or contribute to it: extension authors who
need a new algorithm, problem, or monitor, and contributors who set up the repository,
run the test suite, and write documentation.

```{warning}
This guide documents the **experimental EvoX 2.0 pre-release**, whose import name is
`evox_etl`. EvoX 2.0 replaces the object-oriented 1.x API with plain functions on top of
the [EvoX Compiler](https://github.com/EMI-Group/evoxc) (`evoxc`). The 1.x torch package
(`ModuleBase`, `Parameter`/`Mutable`, `torch.compile`/`vmap`, class-based algorithms,
problems and monitors) was removed in 2.0 — nothing on these pages uses it.
```

| Page | What it covers |
|---|---|
| {doc}`architecture` | How 2.0 fits together: frozen configs, tensor states, the step protocol, and how `StdWorkflow` traces and runs a generation. |
| {doc}`custom-alg-pro` | Write your own algorithm and your own problem, with complete runnable examples. |
| {doc}`custom-monitor` | Write your own monitor, read the run back on the host, and record auxiliary per-generation data. |
| {doc}`environment` | Set up a development environment, run the test suite, and build these docs. |
| {doc}`document` | Documentation conventions: MyST, autodoc, and the bilingual translation workflow. |

```{toctree}
:maxdepth: 1
:hidden:

architecture
custom-alg-pro
custom-monitor
environment
document
```
