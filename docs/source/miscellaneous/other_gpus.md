# Non-NVIDIA Hardware: AMD, Vulkan and Apple Silicon

```{warning} Experimental EvoX 2.0 pre-release
These pages target the **EvoX 2.0 pre-release** (import name `evox_etl`, compiled by `evoxc`).
They do **not** cover the EvoX 1.x torch API (`import evox`, `torch.device("cuda")`,
`device = torch.device("mps")`, `#evox.compile`), which was removed in 2.0 and is documented
nowhere in these pages. APIs may still change between pre-releases.
```

NVIDIA GPUs are the best-trodden path for EvoX: IREE-CUDA and XLA-CUDA were validated on NVIDIA
hardware, and most of the numbers in `REFACTOR_REPORT.md` come from an NVIDIA card (TVM, by contrast, is
CPU-only). That is a statement about testing coverage, not about the programming model — an EvoX 2.0
program never names a vendor. It names a `backend` and a device **kind**, and the framework either serves
that combination or fails loudly.

Which backend serves which device kind, what each kind compiles to, and which extra installs what is
{doc}`../guide/advanced/backends`' subject; this page only records what differs on non-NVIDIA hardware.
In the shorthand: IREE serves `cpu`, `cuda`, `rocm` and `vulkan`, XLA serves `cpu`, `cuda` and `rocm`
once you supply a PJRT plugin, TVM is `cpu` only, and the default NumPy interpreter is `cpu` only. For an
IREE build the compile target is inferred from the device you pass, so changing hardware is one keyword
({doc}`selecting_gpu` has the device forms and how to enumerate what the process can see):

```python
import numpy as np

from evox_etl import EvalMonitorConfig, StdWorkflow
from evox_etl.algorithms import make_pso
from evox_etl.problems.numerical import Sphere

dim = 10
algorithm = make_pso(pop_size=1000, lb=np.full(dim, -100.0), ub=np.full(dim, 100.0))

workflow = StdWorkflow(
    algorithm,
    Sphere(),
    monitor=EvalMonitorConfig(),
    backend="iree",  # needs the IREE extra: pip install "evoxc[iree]"
    device="rocm",   # an AMD GPU; "rocm:1" for the second one
)
workflow.run(100, seed=42)
```

## The CPU path always works

`backend="numpy"` is a pure-Python interpreter over the traced graph: no compiler toolchain, no GPU
driver, no optional package. It runs anywhere CPython and NumPy run — x86-64, ARM64, Windows, macOS on
either Intel or Apple Silicon — and it is the reference implementation the compiled backends are checked
against. IREE results are bit-identical to it for the same seed and configuration; XLA may compute an
fp32 `dot` at TF32-class precision unless you pass `dot_precision="HIGHEST"`, so a dot-heavy graph can
differ in the last bits; TVM is CPU (`llvm`) only. "Develop on numpy, deploy on IREE or XLA" is therefore
a legitimate workflow, not a compromise.

For a faster CPU-only run, two compiled options are available and neither needs a GPU: IREE's
`llvm-cpu` target (`backend="iree", device="cpu"` once the IREE extra is installed) and XLA driven by a
CPU PJRT plugin (`backend="xla", device="cpu"` plus the plugin below).

## AMD GPUs (ROCm)

ROCm is supported end-to-end through IREE's HIP HAL driver, and AMD devices use the ordinary `rocm` device
kind — `"hip"` is accepted as an alias, so `device="hip"` and `device="hip:0"` mean the same thing as
their `rocm` spellings.

```python
workflow = StdWorkflow(
    algorithm,
    Sphere(),
    monitor=EvalMonitorConfig(),
    backend="iree",
    device="rocm",
    compile_options={
        # only needed when the arch cannot be auto-detected (see below)
        "iree_compile_args": ["--iree-rocm-target=gfx942"],
    },
)
```

One ROCm-specific wrinkle: IREE's ROCm target has no arch auto-detection of its own, so the backend
probes for the GPU architecture first (`amdgpu-arch`, then `rocm_agent_enumerator`, on `PATH` or under
`$ROCM_PATH`, `$ROCM_HOME` and `/opt/rocm`) and passes `--iree-rocm-target=<arch>` for you. If both
probes come up empty the build stops with a `BackendError` naming the flag and the remedy — pass the arch
explicitly in `iree_compile_args`, as in the snippet above (`gfx942` is an MI300-class part; use the arch
of your own card). Every IREE default flag can be overridden this way, or through the
`EVOXC_IREE_COMPILE_ARGS` environment variable.

Multi-objective algorithms are the exception here: `iree-compile` cannot currently compile EvoX's
multi-objective step graphs, so on AMD hardware an MO run needs `backend="xla"` with a ROCm PJRT plugin
instead of IREE. {doc}`../guide/advanced/backends` lists the other known limitations.

`evoxc.devices("rocm")` enumerates AMD devices through IREE's HIP HAL driver when IREE is installed, and
through a ROCm build of PyTorch otherwise. This repository's test environment has no AMD hardware, so the
ROCm path is documented here from the code and from the upstream validation runs, not measured locally.

## XLA and the PJRT plugin requirement

`backend="xla"` does not ship a plugin. It drives whatever PJRT C API library you point it at — a `.so`
exporting `GetPjRtApi` — and there is no pip-installable plugin for EvoX to depend on, so the hardware
this backend reaches is decided by the plugin you supply: a CPU plugin keeps it on `device="cpu"`, a
CUDA plugin (`xla_cuda12/`, as a JAX PJRT wheel installs under `jax_plugins/`) makes `device="cuda"` and
`device="cuda:N"` usable, a ROCm plugin (`xla_rocm7/`) makes `device="rocm"` usable — which is also the
route for multi-objective programs on AMD, since IREE cannot compile them.

A plugin comes from an XLA/JAX build or from your own OpenXLA build
(`bazel build //xla/pjrt/c:pjrt_c_api_cpu_plugin`). Name it with the `plugin_path` compile option or the
`EVOXC_PJRT_PLUGIN` environment variable — but export the variable **before the process starts**, because
a cold process's activation probe only consults that variable and the well-known library paths, never a
`plugin_path`. {doc}`../guide/advanced/backends` owns the discovery order, the compile options and the
full plugin warning; from this page only the hardware mapping matters.

With no plugin anywhere, the first build raises an `evoxc.core.BackendError` naming the missing plugin and
the OpenXLA build recipe; it never falls back to another backend.

## Vulkan-capable devices

The only accelerator kind outside `cuda` and `rocm` is `vulkan`, served by IREE's Vulkan HAL driver
(`vulkan-spirv` target). It is the theoretical route to, say, an Intel GPU or an AMD card without a working
ROCm stack, and it is enumerated only on request:

```python
import evoxc

print(evoxc.devices("vulkan"))   # per-kind indices, not Vulkan UUIDs
```

Be aware of two things before relying on it. IREE pins its Vulkan codegen to a baseline profile unless you
pass the `vulkan_target` compile option, so code built for generic Vulkan may not be a good fit for a
specific card — an explicit `vulkan_target` is advisable. And a *software* Vulkan implementation is hidden
by IREE's device-name denylist, so a host whose only Vulkan driver is Mesa's `lavapipe` honestly reports no
devices unless `VK_ICD_FILENAMES` selects a differently named one. This repository has not verified the
Vulkan path end-to-end on any GPU, so treat it as a supported code path rather than a validated setup.

## Apple Silicon

Apple Silicon has no usable accelerator path in this pre-release. `evoxc` knows no `mps` or `metal` device
kind and no backend targets Metal, so the Metal Performance Shaders route that EvoX 1.x offered through
PyTorch does not exist here. `_parse_device` does not reject `device="metal"` — it only splits the string,
so the device comes out as kind `metal` — but no backend serves that kind, so the failure lands when
`init()` places the state on it: with the default NumPy backend this release raises an
`evoxc.core.DeviceError` naming the kind, rather than substituting something else.

What does work on a Mac is the CPU: `backend="numpy"` out of the box, and — with the extras and plugin
above — IREE's `llvm-cpu` target or XLA with a CPU plugin. The programming model is identical; only the
speed differs. This repository's benchmarks run on Linux, so the macOS CPU path is documented from the
code and not measured here. If GPU acceleration on Apple hardware matters to you, that is a feature
request for the `evoxc` compiler backends, not something a workflow argument can switch on.

## When in doubt, make the failure happen early

No backend falls back silently: an unsupported op, a missing compiler or a missing PJRT plugin fails
during the build with an `evoxc.core.BackendError`/`TransformError` naming what is wrong, and a device
kind no backend can serve fails when `init()` places the state on it.
Compilation is paid up front, so build the same workflow with `num_generations=1` (or `run(2)`) on the new
machine before starting a real study — a wiring problem costs a second there and an hour in the middle of a
run.

Continue with {doc}`selecting_gpu` for device selection and multi-GPU pinning, and
{doc}`../guide/advanced/backends` for the backends themselves: install extras, compile options, what each
backend can compile, and the correctness guarantees.
