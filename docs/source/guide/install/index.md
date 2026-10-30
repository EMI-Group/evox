# Installation Guide

```{toctree}
:maxdepth: 1

python
evox
```

This section installs the **experimental EvoX 2.0 pre-release**, the functional rewrite of EvoX whose
import name is `evox`: {doc}`python` prepares a **Python 3.11+** interpreter and an isolated
environment, and {doc}`evox` installs the `evox` distribution — as the `2.0.0a1` pre-release or from a
checkout — which brings the `evoxc` compiler that `import evox` needs, verifies the result with a
short PSO run, and points at the backend guide.
`evoxc` runs the NumPy backend out of the box, so no compiler toolchain or GPU runtime is needed to get
started; the compiled backends are optional extras.

```{warning} Experimental EvoX 2.0 pre-release
These instructions target the **EvoX 2.0 pre-release** and install the `evox` distribution whose import
name is **`evox`**. They do **not** cover the EvoX 1.x torch API (`ModuleBase`,
`Parameter`/`Mutable`, `torch.compile`/`vmap`), which was removed in 2.0 and is documented
nowhere in these pages. APIs may still change between pre-releases.
```
