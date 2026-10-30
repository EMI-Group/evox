# EvoX Installation Guide

EvoX is distributed on PyPI as the **`evox`** package. In the 2.0 pre-release the import name is
**`evox_etl`**, and the framework runs on the [EvoX Compiler](https://github.com/EMI-Group/evoxc)
(`evoxc`), a minimal tensor-graph compiler shipped as a normal Python dependency.
Everything below assumes an activated virtual environment and Python 3.10 or newer — see {doc}`python`
if you have not set one up yet.

## Install from PyPI

```bash
pip install "evox"
```

The default install is pure Python and pulls in the `evoxc` compiler plus NumPy. It runs on the
built-in **NumPy backend**, so no compiler toolchain, GPU driver or PyTorch installation is required;
the compiled backends (IREE, TVM, XLA) are opt-in extras listed below.

```bash
python -c "import evox_etl; print(evox_etl.__version__)"
```

```{note} The 1.x torch implementation is no longer part of the documented surface
EvoX 1.x lived in `src/evox/` as an object-oriented PyTorch API (`import evox`, `ModuleBase`,
`Parameter`/`Mutable`, `torch.compile`/`vmap`) and required `torch >= 2.6.0`. That implementation is
frozen and scheduled for removal: EvoX 2.0 needs no torch, installs none, and the 1.x extras
(`neuroevolution`, `triton`, …) are not part of the documented surface. If you are porting 1.x code,
start from the tutorial rather than from these pages.
```

## Optional extras

Extras can be combined, e.g. `pip install "evox[vis,iree]"`.

| Extra | Installs | Use it for |
|---|---|---|
| *(none)* | `evoxc` + NumPy | the default NumPy backend on CPU: the fastest way to a working install |
| `default` | the curated default feature set (includes the visualization packages) | the safe general-purpose install if you are unsure |
| `vis` | `plotly`, `pandas` | `evox_etl.vis_tools` plotting, `.exv` result files, and `EvalMonitor.plot()` |
| `iree` | `iree-base-compiler` + `iree-base-runtime` (3.11.x) | the `iree` backend — compiled CPU, NVIDIA CUDA or AMD ROCm execution |
| `tvm` | `apache-tvm`, plus `jaxlib` for its MLIR bindings | the `tvm` backend (StableHLO frontend + Relax VM, CPU end-to-end) |
| `compilers` | both compiled toolchains (`iree` + `tvm`) | benchmarking or mixing compiled backends |

```bash
pip install "evox[vis]"             # plotting
pip install "evox[iree]"            # IREE: CPU / CUDA / ROCm
pip install "evox[compilers]"       # IREE + TVM
```

IREE compiles and runs with the same version, so keep `iree-base-compiler` and `iree-base-runtime` in
lockstep — the `iree` extra pins them together.

:::{warning} The XLA backend has no pip extra
`evoxc` drives XLA through the **PJRT C API**, loading a plugin `.so` that *you* provide (for example an
OpenXLA build of `pjrt_c_api_cpu_plugin`). No PyPI package ships that binary, so no extra can enable the
backend. Point the compiler at the plugin with an environment variable:

```bash
export EVOXC_PJRT_PLUGIN=/path/to/pjrt_c_api_cpu_plugin.so
```

Or pass it per workflow as a compile option:
`StdWorkflow(..., backend="xla", compile_options={"plugin_path": "..."})`.
With neither set, `evoxc` probes a few well-known library paths and then fails with a `BackendError`
that names the missing plugin.
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
print(workflow.monitor.get_best_fitness())  # ~0.03 and falling; the Sphere optimum is 0.0
```

A printed number near zero — not an exception — means the framework, the compiler and the problem are
wired up correctly: `run()` builds the step graph through `evoxc.build`, executes 50 generations, and
leaves the best fitness on the host-side monitor. With `backend=` at its default, this needs nothing but
Python and NumPy.

## Install a development checkout

To work on EvoX itself, install an editable checkout so that edits under `src/` take effect immediately:

```bash
git clone https://github.com/EMI-Group/evox.git
cd evox
pip install -e .
```

With `uv`, run `uv pip install -e .` inside an activated environment instead.
The checkout is pure Python, so an editable install needs no build step. See
{doc}`../developer/environment` for the full contributor setup (test extras, repository layout,
running the suites).

## Next steps

- {doc}`../../tutorial/tutorial_part2` — walkthrough of the programming model, starting from an install.
- {doc}`../advanced/backends` — which backend to run on, and what each one supports.
