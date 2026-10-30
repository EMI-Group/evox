# Installation Guide

```{toctree}
:maxdepth: 1

python
evox
```

This section installs the **experimental EvoX 2.0 pre-release**, the functional rewrite of EvoX whose
import name is `evox_etl`: {doc}`python` prepares a Python interpreter and an isolated environment, and
{doc}`evox` installs the `evox` distribution, verifies it with a short PSO run, and points at the backend
guide. The default install is pure Python and runs on the built-in NumPy backend, so no compiler
toolchain or GPU runtime is needed to get started.

```{warning} Experimental EvoX 2.0 pre-release
These instructions target the **EvoX 2.0 pre-release** and install the `evox` distribution whose import
name is **`evox_etl`**. They do **not** cover the EvoX 1.x torch API (`import evox`, `ModuleBase`,
`Parameter`/`Mutable`, `torch.compile`/`vmap`), which is frozen, scheduled for removal, and documented
nowhere in these pages. APIs may still change between pre-releases.
```
