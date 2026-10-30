# Backends and Performance

```{warning} Experimental EvoX 2.0 pre-release
These pages target the **EvoX 2.0 pre-release** (import name `evox_etl`, compiled by `evoxc`).
They do **not** cover the EvoX 1.x torch API (`import evox`, `ModuleBase`, `Parameter`/`Mutable`,
`torch.compile`/`vmap`), which is frozen, scheduled for removal, and documented nowhere in these
pages. APIs may still change between pre-releases.
```

A **backend** is where an EvoX program actually runs. It is *not* a global setting and there is no
environment switch to flip before `import`: it is the `backend=` argument of the `StdWorkflow` owning
the compiled step graph.

```python
from evox_etl import EvalMonitorConfig, StdWorkflow

workflow = StdWorkflow(algorithm, problem, monitor=EvalMonitorConfig(), backend="numpy")  # default
```

Both defaults are unexciting: `backend="numpy"` and `device=None`, which means CPU. Algorithm, problem
and monitor configs are backend-agnostic, so switching backend changes one keyword, not the program.
Each step *variant* (`init_step` / `step` / `final_step`) is traced and compiled **once** by
`evoxc.build` (cached per state shape signature), then run generation after generation — re-evaluating
per generation is the anti-pattern the workflow exists to avoid.

```python
# src/evox_etl/core/workflow.py — StdWorkflow._build_step_exe (abridged)
options = {k: v for k, v in self._compile_options.items() if k not in ("backend", "device")}
return evoxc.build(step_fn, specs, backend=self.backend, device=self._device, **options)
```

The algorithm-init and monitor-init graphs are *always* built on the numpy backend and the CPU
(`evoxc.build(..., backend="numpy", device=evoxc.core.Device("cpu"))`); only step graphs chase speed.

## The NumPy backend

`"numpy"` is a pure-Python interpreter over the traced graph: an op table driven over the IR, with
`if`/`while`/`scan` bodies executed by recursively interpreting their region blocks. It needs no
compiler toolchain, no GPU driver and no optional package. It is *not* an eager sandbox, though — EvoX
2.0 has no eager mode, so a tensor op outside a trace raises `evoxc.core.TraceError` even here. What
you get instead is a debuggable failure: a Python stack trace inside *your* traced function, with real
shapes and dtypes. It is also the reference implementation and slow at scale, so use it for first runs,
tests, and any workload small enough that compiling costs more than it saves ({doc}`../install/evox`).

## The compiled backends

| Backend | Install | Targets | State |
|---|---|---|---|
| `"numpy"` | nothing (default) | CPU | always available |
| `"iree"` | `pip install "evox[iree]"` (`iree-base-compiler` + `iree-base-runtime`) | `llvm-cpu`, `cuda`, `rocm`, `vulkan-spirv` | validated end-to-end |
| `"tvm"` | `pip install "evox[tvm]"` (`apache-tvm` + `jaxlib` for its MLIR bindings) | `llvm` CPU only | limited; see below |
| `"xla"` | **no extra** — you supply a PJRT plugin `.so` | CPU, CUDA, ROCm | validated end-to-end |

`evox[compilers]` installs IREE and TVM together; `evox[vis]` is unrelated (Plotly/`.exv` output).

**IREE** compiles the StableHLO export with `iree-compile` and runs it through the IREE runtime: the
`local-task` CPU driver, or the `cuda`/`hip`/`vulkan` HAL drivers on GPUs. Compiler and runtime must
stay version-locked, which is why the `iree` extra pins the two `iree-base-*` packages together.

**TVM** translates StableHLO into a Relax module and builds it for `target=llvm`, so it is CPU-only.
Its op whitelist excludes control flow (`stablehlo.if`/`stablehlo.while`) and `gather`/`scatter`, which
are rejected at compile time with a `BackendError` naming the op.

**XLA** is driven through the **PJRT C API** by loading a plugin `.so` (exporting `GetPjRtApi`) that
*you* provide — no PyPI package ships one, so no extra can enable the backend. Name it with
`export EVOXC_PJRT_PLUGIN=/path/to/pjrt_c_api_cpu_plugin.so`, or per workflow with the `plugin_path`
compile option (discovery order: `plugin_path`, then `EVOXC_PJRT_PLUGIN`, then a few well-known
library paths). Be aware that the backend is activated on first use, and that probe only consults
`EVOXC_PJRT_PLUGIN` and the well-known paths — on a cold process a `plugin_path` alone is not enough,
so set the environment variable too. With no plugin anywhere, the first build fails with an
`evoxc.core.BackendError` naming the missing plugin and the OpenXLA `bazel build` recipe.

## How `backend=`, `device=` and `compile_options=` reach the compiler

`backend` and `device` are consumed by the workflow; everything else in `compile_options` is forwarded
as keyword options to `evoxc.build`, which hands each option to the pipeline stage that understands it
(lower / compile / load) and rejects a key no stage understands. `backend` and `device` are stripped
out of `compile_options` first, so passing them there is silently harmless rather than a double
definition.

| Stage | Options |
|---|---|
| lower (exporter) | `dot_precision`, `eigh_algorithm`, `split_reduction`, `rng_bit_generator`, plus `sort_emission` and `while_init_rewrite` on IREE |
| compile | iree: `target_backends`, `iree_compile_args`, `opt_level`, `vulkan_target` · xla: `plugin_path`, `xla_compile_options`, `opt_level` · tvm: `tvm_target`, `tvm_pass_configs`, `opt_level` |
| load | iree: `iree_runtime_args` · xla: `plugin_path` |

`device=` accepts three forms, parsed by `_parse_device` before anything is built:

| `device=` | Result | Rule |
|---|---|---|
| `None` | `Device("cpu", 0)` | default |
| `"cpu"`, `"cuda"`, `"rocm"`, `"vulkan"` | `Device(kind, 0)` | string is stripped and lower-cased |
| `"cuda:1"` | `Device("cuda", 1)` | split on `:`, the index is converted with `int()` |
| `evoxc.core.Device("rocm", 0)` | unchanged | passed through as-is |
| anything else | `TypeError` | |

Supported kinds are `cpu`, `cuda`, `rocm` and `vulkan`; `"hip"` is accepted as an alias/equivalent
filter for the `rocm` kind by device enumeration. `_parse_device` only splits the string — a kind the
chosen backend cannot serve fails loudly when the executable loads, never by silently falling back to
CPU.

```{warning} `device=` does not create acceleration on its own
The NumPy backend runs on the CPU only, and asking it for a GPU device does not quietly stay on the
CPU: the workflow tries to move the data and, with no compiled backend installed, fails in the transfer
path. Pick a backend that serves the device, e.g. `backend="iree", device="cuda"`.
```

## Choosing a backend

| Workload | Use | Why |
|---|---|---|
| small populations / low dimension, first run, debugging | `numpy` | zero install, and failures point at your traced Python |
| small/medium workloads on a CPU host, long runs or many seeds | `iree` (`llvm-cpu`) or `xla` (CPU plugin) | one compile, then native execution instead of interpretation |
| medium populations on a GPU | `iree` (`cuda`/`rocm`) or `xla` | keeps the step on the device |
| large single-objective batches (10⁴ × 10²) | `xla` (`cuda`) | the fastest evoxc option there; see the numbers below |
| multi-objective (NSGA2/NSGA3/MOEAD/RVEA/…) | `xla` (`cuda`) | **IREE cannot compile MO programs** |
| loop-free, CPU-only, minimal toolchain | `tvm` | llvm CPU target; no control flow |

Numpy stays the default on purpose: a compiled backend needs an extra (IREE, TVM) or a plugin you build
yourself (XLA), and compilation is a fixed cost paid per step variant that dominates at small scale.

## Known user-visible limitations

- **IREE cannot compile multi-objective programs.** `iree-compile` segfaults (`-11`) on every MO step
  graph — an upstream IREE while-loop bug, reproducible with an unmodified NSGA2 — and one MOEAD/DTLZ2
  cell additionally failed at run time on `iree-llvm-cpu` (`ref is null`). Use XLA for MO.
- **IREE CUDA forbids `opt_level` `O2`/`O3`** (codegen bugs on `eigh` and NSGA2-selection graphs).
  Leave the defaults: `while_init_rewrite` on and `sort_emission` `"auto"` (`"count"` on CUDA targets).
- **Large-scale CMA-ES is slow when compiled** — the compiled while-loop is the cost. It is correct
  everywhere, just expensive at 10⁴ × 10² (~22 s/step on `iree-llvm-cpu`).
- **The StableHLO exporter defers `svd`/`matrix_rank`/`solve`/`cumprod`/`flip`.** NSGA3 and DTLZ are
  worked around inside EvoX (eigh-based rank, `reduce_prod`, `gather`); ASEBO's `step` still calls
  `evoxc.svd`, so ASEBO runs on the numpy backend only.
- **TVM has no control flow**, so NSGA-family generation loops do not compile on it.
- **XLA dynamic shapes are unusable.** The adapter's static-shape gate raises `BackendError` naming the
  offending spec instead of miscompiling it.
- **Failures are loud.** Unsupported ops, devices or options raise `evoxc.core.BackendError` or
  `TransformError` (unknown backend name, missing IREE install with its pip hint, unsupported device
  kind); there is never a silent fallback to another backend.

## Correctness

Fitnesses produced by the four evoxc backends (numpy, IREE, TVM, XLA) are **bit-identical** for the same
seed and configuration: a backend changes *how fast* a run is, not *what it finds* — which is why you can
debug on numpy and deploy on XLA. The one caveat sits outside the backends: on an unconverged stochastic
problem, two *different* runs scatter anyway.

The numbers on this page are workload-dependent measurements from `benchmarks/etl_vs_torch/`
(RTX A6000, seed 42, 100 generations), summarized in `REFACTOR_REPORT.md` §3: at 10⁴ × 10²,
XLA-on-CUDA runs PSO/DE/OpenES in 8–18 ms/step (torch-cpu 8–36 ms/step, IREE-on-CUDA ~3.5 s/step);
on the MO matrix XLA-on-CUDA runs all 12 cases at a median **28× faster than torch-cpu**; large-scale
CMA-ES is the slowest evoxc cell. A hand-written CUDA kernel stack (the frozen 1.x torch build) is still
faster at the very largest single-objective batches — PSO ~1.1 vs XLA-on-CUDA ~17 ms/step — so the gain
is portability across CPU/CUDA/ROCm from one program, not peak kernel throughput.

## A runnable example

This program is complete and uses the default backend, so it needs nothing but Python and NumPy.
`lb`/`ub` must be 1-D arrays (or flat tuples) of length `dim`; nested or 2-D bounds are rejected.

```python
import numpy as np

from evox_etl import EvalMonitorConfig, StdWorkflow
from evox_etl.algorithms import PSO
from evox_etl.problems.numerical import Sphere

dim = 10
algorithm = PSO(pop_size=100, lb=np.full(dim, -100.0), ub=np.full(dim, 100.0))
problem = Sphere()

workflow = StdWorkflow(algorithm, problem, monitor=EvalMonitorConfig(), backend="numpy")
workflow.run(100, seed=42)  # 100 generations, seed 42
print(workflow.backend, f"{workflow.monitor.get_best_fitness():.3e}")
```

```text
numpy 3.054e-08
```

Sphere is minimized at the origin, so a value near zero is the correct answer. Nothing above depends on
the backend — move it to a compiler by changing those arguments and installing what the backend needs:

```python
# pip install "evox[iree]"   ->  compiled CPU / CUDA / ROCm via IREE
workflow = StdWorkflow(algorithm, problem, monitor=EvalMonitorConfig(),
                       backend="iree",
                       device="cuda:1")   # parsed to evoxc.core.Device(kind="cuda", index=1)

# XLA: no extra; the PJRT plugin is yours. Keep the env var set and/or pin it per workflow.
workflow = StdWorkflow(algorithm, problem, monitor=EvalMonitorConfig(),
                       backend="xla", device="cpu",
                       compile_options={"plugin_path": "/path/to/pjrt_c_api_cpu_plugin.so"})
```

```{tip} Prove the backend with a tiny run first
Compilation is paid up front, so validate the wiring before committing an hour: build the same workflow
with `num_generations=1` (or `run(2)`) and check that it returns instead of raising — a `BackendError`
or `TransformError` there is a capability gap, much cheaper to read than in the middle of a study.
```

Continue with {doc}`operators` and {doc}`metrics` for the pieces that feed and score these runs,
{doc}`../developer/custom-alg-pro` for writing your own algorithm, and {doc}`../../tutorial/tutorial_part3`
for the workflow contract in full.
