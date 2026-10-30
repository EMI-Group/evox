# EvoX Installation Guide

EvoX is distributed on PyPI as the **`evox`** package. In the 2.0 pre-release the import name is
**`evox_etl`**, and the framework runs on the [EvoX Compiler](https://github.com/EMI-Group/evoxc)
(`evoxc`), a minimal tensor-graph compiler shipped as a normal Python dependency.
Everything below assumes an activated virtual environment and **Python 3.11 or newer** — see
{doc}`python` if you have not set one up yet.

## Install from a source checkout

EvoX 2.0 has no published wheel yet: the `evox` release on PyPI is still the 1.x line, whose tree does
not contain `src/evox_etl`. Install from a checkout, and install the compiler next to it:

```bash
git clone https://github.com/EMI-Group/evox.git
cd evox
pip install -e .        # the `evox` distribution — its metadata is still the 1.x manifest
pip install evoxc       # the compiler that `import evox_etl` needs
```

The second command is not optional, and the first one is not enough on its own. `evox`'s packaging is
still the 1.x manifest: it declares `torch >= 2.6.0` and `numpy >= 2.0.0`, and it does **not** declare
`evoxc`. So `pip install -e .` pulls PyTorch (which no 2.0 code path imports) and leaves the compiler
out, while `import evox_etl` imports `evoxc` unconditionally — an environment with only `evox` fails
with `ModuleNotFoundError: No module named 'evoxc'`.

Check that both halves are importable:

```bash
python -c "import evox_etl"
```

```{note} Verify with an import, not a version number
`evox_etl.__version__` is inherited from the 1.x line and does not track the 2.0 pre-release, so it is
not a usable 2.0 check. Importing the package (above) and running the smoke test further down are.
```

```{note} The 1.x torch implementation is no longer part of the documented surface
EvoX 1.x lived in `src/evox/` as an object-oriented PyTorch API (`import evox`, `ModuleBase`,
`Parameter`/`Mutable`, `torch.compile`/`vmap`) and required `torch >= 2.6.0`. That implementation is
frozen and scheduled for removal, and none of it — nor its extras — is part of the documented surface.
If you are porting 1.x code, start from the tutorial rather than from these pages.
```

## Optional extras

The install above already runs on the NumPy backend; everything else is optional, and extras can be
combined, e.g. `pip install "evox[vis]" "evoxc[iree]"`.

The compiled backends come from the compiler's extras. EvoX documents them as extras of the `evox`
distribution — `evox[iree]`, `evox[tvm]`, `evox[compilers]` — and the compiler package exposes the same
three as `evoxc[iree]`, `evoxc[tvm]` and `evoxc[compilers]`, which is what you install when you manage
`evoxc` separately.

| Extra | Installs | Use it for |
|---|---|---|
| *(none)* | `evox` itself, with `torch >= 2.6.0` + `numpy >= 2.0.0` from its 1.x manifest; **not** `evoxc` | the smallest base; add `pip install evoxc` to get a working `import evox_etl` |
| `vis` | `plotly`, `pandas` | `evox_etl.vis_tools` plotting, `.exv` result files, and `EvalMonitor.plot()` |
| `iree` (or `evoxc[iree]`) | `iree-base-compiler` + `iree-base-runtime` (3.11.x) | the `iree` backend — compiled CPU, NVIDIA CUDA or AMD ROCm execution |
| `tvm` (or `evoxc[tvm]`) | `apache-tvm`, plus `jaxlib` for its MLIR bindings | the `tvm` backend (StableHLO frontend + Relax VM, CPU end-to-end) |
| `compilers` (or `evoxc[compilers]`) | both compiled toolchains (`iree` + `tvm`) | benchmarking or mixing compiled backends |

```bash
pip install "evox[vis]"             # plotting
pip install "evox[iree]"            # IREE: CPU / CUDA / ROCm
pip install "evoxc[iree]"           # the same IREE toolchain, on the compiler directly
pip install "evox[compilers]"       # IREE + TVM
```

```{note} `evox[...]` versus `evoxc[...]`
`evox`'s `optional-dependencies` table is still the 1.x manifest, which declares neither `evoxc` nor a
compiled-backend extra: today `pip install "evox[iree]"` warns that the extra does not exist and
installs nothing, while `pip install "evoxc[iree]"` really does pull the toolchain. The `evox[...]`
spellings are the intended 2.0 surface; until the manifest catches up, use the `evoxc[...]` spelling to
get the compiler extra installed.
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
