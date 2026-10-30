# EvoX Installation Guide

EvoX is distributed on PyPI as the **`evox`** package. In the 2.0 pre-release the import name is
**`evox_etl`**, and the framework runs on the [EvoX Compiler](https://github.com/EMI-Group/evoxc)
(`evoxc`), a minimal tensor-graph compiler shipped as a normal Python dependency.
Everything below assumes an activated virtual environment and **Python 3.11 or newer** — see
{doc}`python` if you have not set one up yet.

## Install the 2.0 pre-release

The distribution name is still **`evox`**, and the 2.0 line ships as **pre-releases** — `2.0.0a1` is
the current one. PyPI derives pre-release status from the version string itself, so a plain
`pip install evox` keeps resolving the **1.x stable line**, whose tree has no `src/evox_etl`. Ask for
the pre-release explicitly:

```bash
pip install --pre evox==2.0.0a1     # this pre-release, exactly
pip install "evox>=2.0.0a1"         # newest 2.0 pre-release; the specifier itself opts into pre-releases
```

Either command installs the framework **and** the compiler it runs on: `pyproject.toml` declares
`dependencies = ["evoxc >= 0.1.0", "numpy >= 2.0.0"]` and `requires-python = ">=3.11"`, so nothing else
has to be installed by hand.
If no 2.0 pre-release is on the index yet, both commands fail with
`No matching distribution found for evox==2.0.0a1` — use the source checkout below.

## Install from a source checkout

```bash
git clone https://github.com/EMI-Group/evox.git
cd evox
pip install -e .
```

One command is enough: the editable install of the `evox` distribution pulls `evoxc >= 0.1.0` (the
compiler that `import evox_etl` needs) and `numpy >= 2.0.0` as declared dependencies. It installs no
PyTorch — no 2.0 code path imports it.

Check that the package is importable:

```bash
python -c "import evox_etl"
```

```{note} Where `evox_etl.__version__` comes from
The module reads it from the installed distribution metadata (`importlib.metadata.version("evox")`), so
it reports `2.0.0a1` once the `evox` distribution is installed, and `0.0.0.dev0` in a bare source
checkout that was never installed (expected there, not a broken environment). The import above and the
smoke test further down remain the checks that matter.
```

```{note} The 1.x torch implementation is no longer part of this tree
EvoX 1.x lived in `src/evox/` as an object-oriented PyTorch API (`import evox`, `ModuleBase`,
`Parameter`/`Mutable`, `torch.compile`/`vmap`) and required `torch >= 2.6.0`. It was removed in 2.0:
`src/evox/` does not exist any more, and no 2.0 code path imports torch. If you still run 1.x code, keep
it in an environment pinned to `evox<2`; {doc}`../../tutorial/tutorial_part7` maps each removed API to
its 2.0 equivalent.
```

## Optional extras

The install above already runs on the NumPy backend; everything else is optional, and extras can be
combined, e.g. `pip install "evox[vis]" "evox[iree]"`.

The compiled backends come from the compiler's extras, which the `evox` metadata re-exports: `evox[iree]`
is declared as `evoxc[iree]`, and the compiler package exposes the same three extras directly as
`evoxc[iree]`, `evoxc[tvm]` and `evoxc[compilers]` — the spelling to use when you manage `evoxc` yourself.

| Extra | Installs | Use it for |
|---|---|---|
| *(none)* | `evox` plus `evoxc >= 0.1.0` and `numpy >= 2.0.0` | the smallest base — a complete, working install on the NumPy backend |
| `vis` | `plotly`, `pandas` | `evox_etl.vis_tools` plotting, `.exv` result files, and `EvalMonitor.plot()` |
| `iree` (or `evoxc[iree]`) | `iree-base-compiler` + `iree-base-runtime` (3.11.x) | the `iree` backend — compiled CPU, NVIDIA CUDA or AMD ROCm execution |
| `tvm` (or `evoxc[tvm]`) | `apache-tvm`, plus `jaxlib` for its MLIR bindings | the `tvm` backend (StableHLO frontend + Relax VM, CPU end-to-end) |
| `compilers` (or `evoxc[compilers]`) | both compiled toolchains (`iree` + `tvm`) | benchmarking or mixing compiled backends |
| `all` | `evoxc[compilers]` + `plotly` + `pandas` | every optional runtime dependency in one command |

`default` survives from 1.x as an alias of `vis`, and `test` / `docs` are the developer extras (pytest
plus plotting; the Sphinx toolchain) — {doc}`../developer/environment` uses them.

```bash
pip install "evox[vis]"             # plotting
pip install "evox[iree]"            # IREE: CPU / CUDA / ROCm
pip install "evoxc[iree]"           # the same IREE toolchain, on the compiler directly
pip install "evox[compilers]"       # IREE + TVM
```

```{note} `evox[...]` versus `evoxc[...]`
Both spellings install the same toolchain. `pip install "evox[iree]"` resolves to
`evoxc[iree] >= 0.1.0` from the `evox` metadata and really does pull `iree-base-compiler` +
`iree-base-runtime`; `pip install "evoxc[iree]"` installs the identical packages when the compiler is
managed as its own distribution. Use one of them, so both cannot disagree about the compiler version.
```

IREE compiles and runs with the same version, so keep `iree-base-compiler` and `iree-base-runtime` in
lockstep — the `iree` extra pins them together. The extra also installs no Vulkan loader or ICD, so the
`vulkan-spirv` target additionally needs a Vulkan driver on the host (Mesa `lavapipe` works for
testing), and without a `vulkan_target` profile IREE's default profile caps workgroup shared memory at
16 KiB — large graphs fail to compile there.

:::{warning} The XLA backend has no pip extra
`evoxc` drives XLA through the **PJRT C API**, loading a plugin `.so` that *you* provide (for example an
OpenXLA build of `pjrt_c_api_cpu_plugin`). No PyPI package ships that binary, so no extra can enable the
backend. Export the plugin path before you run the program:

```bash
export EVOXC_PJRT_PLUGIN=/path/to/pjrt_c_api_cpu_plugin.so
```

On a cold process that variable is how the plugin is found: it is the only input besides a few
well-known library paths. The `plugin_path` compile option
(`StdWorkflow(..., backend="xla", compile_options={"plugin_path": "..."})`) is not a standalone
alternative — it is honored by the compile stage, but loading and running the executable rediscovers the
plugin from the environment, so the variable has to be set as well. With no plugin anywhere, the first
build fails with a `BackendError` that names the missing plugin. {doc}`../advanced/backends` has the
full discovery order.
:::

## Choosing a backend

`StdWorkflow` selects the backend and device per workflow, not at import time. `backend=` defaults to
`"numpy"` and `device=` defaults to `"cpu"`; `device` also accepts `"cuda"`, `"cuda:1"`, `"rocm"`,
`"hip"` and `"vulkan"`, where the backend and the hardware support them.

```python
workflow = StdWorkflow(algorithm, problem, monitor=EvalMonitorConfig(), backend="iree", device="cuda")
```

Capabilities differ per backend — read {doc}`../advanced/backends` before moving a workload.

## Verify the installation

The quickest end-to-end check is a 50-generation PSO run on the Sphere problem. Save this as `smoke.py`
and run it with `python smoke.py`:

```python
from evox_etl import StdWorkflow, EvalMonitorConfig
from evox_etl.algorithms import PSO
from evox_etl.problems import Sphere

workflow = StdWorkflow(PSO(pop_size=100, lb=[-100.0] * 10, ub=[100.0] * 10), Sphere(), monitor=EvalMonitorConfig())
workflow.run(50, seed=42)
print(workflow.monitor.get_best_fitness())  # a small positive number and falling; the Sphere optimum is 0.0
```

```console
$ python smoke.py
0.027914267033338547
```

A printed number near zero — not an exception — means the framework, the compiler and the problem are
wired up correctly: `run()` builds the step graph through `evoxc.build`, executes 50 generations, and
leaves the best fitness on the host-side monitor. With `backend=` at its default, this needs `evoxc` and
NumPy, nothing more.

## Work on EvoX itself

The checkout you cloned above is already installed in editable mode, so edits under `src/` take effect
immediately — there is no separate build step. With `uv`, run `uv pip install -e .` inside an activated
environment instead. See {doc}`../developer/environment` for the full contributor setup (test extras,
repository layout, running the suites).

## Next steps

- {doc}`../../tutorial/tutorial_part2` — walkthrough of the programming model, starting from an install.
- {doc}`../advanced/backends` — which backend to run on, and what each one supports.
