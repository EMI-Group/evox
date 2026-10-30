# Installation Guide

```{toctree}
:maxdepth: 1

python
evox
```

This section installs the **experimental EvoX 2.0 pre-release**, the functional rewrite of EvoX whose
import name is `evox_etl`: {doc}`python` prepares a **Python 3.11+** interpreter and an isolated
environment, and {doc}`evox` installs the `evox` distribution from a checkout, adds the `evoxc` compiler
that `import evox_etl` needs, verifies the result with a short PSO run, and points at the backend guide.
`evoxc` runs the NumPy backend out of the box, so no compiler toolchain or GPU runtime is needed to get
started; the compiled backends are optional extras.

```{warning} Experimental EvoX 2.0 pre-release
These instructions target the **EvoX 2.0 pre-release** and install the `evox` distribution whose import
name is **`evox_etl`**. They do **not** cover the EvoX 1.x torch API (`import evox`, `ModuleBase`,
`Parameter`/`Mutable`, `torch.compile`/`vmap`), which was removed in 2.0 and is documented
nowhere in these pages. APIs may still change between pre-releases.
```
