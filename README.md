<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/source/_static/evox_brand_light.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/source/_static/evox_brand_dark.svg">
    <img alt="EvoX Logo" height="128" width="500px" src="docs/source/_static/evox_brand_dark.svg">
  </picture>
</h1>

## ⚠️ Experimental Pre-Release (2.0.0a1)

> [!WARNING]
> This branch is the **experimental EvoX 2.0 pre-release (`2.0.0a1`)** — a from-scratch **functional rewrite** of
> EvoX on top of the [**EvoX Compiler**](https://github.com/EMI-Group/evoxc) (`evoxc`) tensor library.
> It is **not** the 1.x PyTorch API: the package is imported as **`evox_etl`** (there is no `import evox` module in
> 2.0.0a1), and the 1.x PyTorch implementation has been deleted from this branch.
> The distribution name is still **`evox`**, so PyPI treats `2.0.0a1` as a **pre-release**: a plain
> `pip install evox` keeps giving you the stable **1.x** line, which is the recommended choice for most users.
> Only `pip install --pre evox==2.0.0a1` (or `pip install 'evox>=2.0.0a1'`) installs the new line — do that in a
> separate virtual environment, because both lines share the `evox` distribution name and cannot coexist.
> APIs may still change between pre-releases.

<div align="center">
  <a href="https://arxiv.org/abs/2301.12457"><img src="https://img.shields.io/badge/arxiv-2212.05652-red" alt="arXiv"></a>
  <a href="https://evox.readthedocs.io/en/latest/index.html"><img src="https://img.shields.io/badge/readthedocs-docs-green?logo=readthedocs" alt="Documentation"></a>
  <a href="https://pypi.org/project/evox/"><img src="https://img.shields.io/pypi/v/evox?logo=python" alt="PyPI Version"></a>
  <a href="https://pypi.org/project/evox/"><img src="https://img.shields.io/badge/python-3.11+-orange?logo=python" alt="Python Version"></a>
  <a href="https://discord.gg/Vbtgcpy7G4"><img src="https://img.shields.io/badge/discord-evox-%235865f2?logo=discord" alt="Discord Server"></a>
  <a href="https://qm.qq.com/q/vTPvoMUGAw"><img src="https://img.shields.io/badge/QQ-297969717-%231db4f4?logo=tencentqq" alt="QQ Group"></a>
</div>

<p align="center">
  <a href="./README.md"><img src="https://img.shields.io/badge/English-f6f5f4" alt="English README"></a>
  <a href="./README_ZH.md"><img src="https://img.shields.io/badge/中文-f6f5f4" alt="中文 README"></a>
</p>

---

<h3 align="center"> 🌟A Functional, Compiler-Backed Framework for Scalable Evolutionary Computation🌟 </h3>

---

## 🔥 News
- [2026-10-30] Released **EvoX 2.0.0a1** (experimental pre-release) — EvoX is now a from-scratch **functional** rewrite on the **EvoX Compiler** (`evoxc`), imported as `evox_etl`. [[Release Notes](https://github.com/EMI-Group/evox/releases/tag/v2.0.0a1)]
- [2026-09-09] Released **EvoX 1.4.0** - 🚀 A maintenance release with various fixes and improvements! [[Release Notes](https://github.com/EMI-Group/evox/releases/tag/v1.4.0)]

## Table of Contents

1. [Experimental Pre-Release (2.0.0a1)](#-experimental-pre-release-2000a1)
2. [Overview](#overview)
3. [Key Features](#key-features)
4. [Main Contents](#main-contents)
5. [Installation](#installation)
6. [Quick Start](#quick-start)
7. [Programming Model](#programming-model)
8. [What Changed Since 1.x](#what-changed-since-1x)
9. [Documentation](#documentation)
10. [Sister Projects](#sister-projects)
11. [Community & Support](#community--support)
12. [Citing EvoX](#citing-evox)

## Overview

EvoX is a GPU-accelerated framework for scalable evolutionary computation. With a user-friendly programming model, it offers a comprehensive suite of **140+ Evolutionary Algorithms (EAs)** and **240+ Benchmark Problems/Environments** across the EvoX ecosystem — **30+ algorithms** and **30+ benchmark problems** built into this package, plus its sister libraries. For more details, please refer to our [Paper](https://arxiv.org/abs/2301.12457) and [Documentation](https://evox.readthedocs.io/en/latest/index.html) / [文档](https://evox.readthedocs.io/zh-cn/latest/index.html).

EvoX 2.0 keeps that catalogue but changes the machinery: every algorithm, problem, monitor and metric is plain
Python — frozen configuration dataclasses, tensor-valued state, and `init`/`step` functions — and the component
model's only compile site is `StdWorkflow`, which traces one step graph through `evoxc` and runs it in a loop.
The same program therefore runs on a pure-NumPy interpreter or on a compiled IREE / TVM / XLA backend for CPU and
GPU by changing one argument.

> [!NOTE]
> Looking for the stable release? `pip install evox` still resolves to the **1.x** PyTorch line, whose JAX-based
> predecessor lives on the **v0.9.0** branch. This repository's default branch is the 2.0 pre-release.

## Key Features

### 🚀 Compiled, Not Interpreted

- **One program, several execution modes**: `backend="numpy"` (default, no toolchain needed), `"iree"` (compiled
  CPU / CUDA / ROCm), `"tvm"` (compiled CPU) or `"xla"` (PJRT). Only `StdWorkflow(backend=..., device=...)` changes;
  algorithm and problem configs stay identical.
- **Compile once, loop on the device**: `StdWorkflow` builds the step graph a single time and feeds outputs back as
  inputs, instead of re-tracing per generation.
- **GPU-class throughput**: in the committed harness `benchmarks/etl_vs_torch/`, the compiled XLA-CUDA backend runs
  the same PSO/DE cells **1.16–1.45× faster per step than the 1.x torch implementation** on an RTX A6000, at
  identical convergence ([BENCHMARK_RESULTS.md](benchmarks/etl_vs_torch/BENCHMARK_RESULTS.md)).

### 🌐 All-in-One Solution

- **32 distinct algorithm classes** (27 single-objective names, one of which — `VirtualLoRAES` — aliases `VirtualES`, plus 6 multi-objective names) covering DE, ES, PSO, the training-free virtual ES family, NSGA-II/III, MOEA/D, RVEA, RVEA-a and HypE, with **140+ algorithms** available across the EvoX ecosystem.
- **30+ benchmark problems**: 10 basic numerical problems (9 classic functions plus a shift/affine wrapper), DTLZ1–7,
  all 12 CEC2022 functions, and virtual (Gaussian-noise neuroevolution) problems, with **240+ benchmark
  problems/environments** across the ecosystem.
- **Host-side hyperparameter optimization** through `evox_etl.problems.hpo_wrapper` (see
  [What Changed Since 1.x](#what-changed-since-1x) for why it is host-side).

### 🛠️ Easy-to-Use Functional Design

- **Configuration is data**: every component is a frozen dataclass, usually built by a `make_*` constructor
  (`make_pso(...)`, `make_cma_es(...)`, `make_nsga2(...)`).
- **State is data**: `init(config, key)` returns a pytree of tensors, `step(config, state, evaluate)` returns the
  next one — serialize it, diff it, replay it.
- **Explicit randomness**: stateless keys (`evox_etl.key(seed)`) live inside the state, so a single integer seed
  reproduces a whole run; there is no global seed.
- **Plain functions everywhere**: algorithms, operators and metrics are ordinary Python you can test and reuse —
  no framework base class to inherit.

### 📊 Flexible Visualization & Metrics

- **Ready-to-use analysis**: `EvalMonitor` records fitness, elite and population histories plus auxiliary channels,
  and `EvalMonitor.plot()` returns an interactive Plotly figure.
- **Multi-objective metrics**: `gd` and `igd` (with `gd_plus`/`igd_plus`) plus `hv` in `evox_etl.metrics`.
- **Real-time data streaming**: the tailored **`.exv`** format (`evox_etl.vis_tools`) serializes runs for fast,
  incremental loading.

## Main Contents

<table border="1" cellspacing="0" cellpadding="8" style="border-collapse: collapse; width: 100%; text-align: left;">
  <thead>
    <tr style="background-color: #f2f2f2;">
      <th>Category</th>
      <th>Subcategory</th>
      <th>Algorithms / Benchmark Problems in this package</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3">Single-objective Optimization</td>
      <td><b>Differential Evolution</b></td>
      <td>DE, CoDE, JaDE, SaDE, SHADE, ODE</td>
    </tr>
    <tr>
      <td><b>Evolution Strategy</b></td>
      <td>CMA-ES, OpenES, xNES, SNES, SeparableNES, ARS, ASEBO, DES, ESMC, GuidedES, NoiseReuseES, PersistentES, VirtualES / VirtualLoRAES</td>
    </tr>
    <tr>
      <td><b>Particle Swarm Optimization</b></td>
      <td>PSO, CLPSO, CSO, DMS-PSO-EL, FS-PSO, SL-PSO-GS, SL-PSO-US</td>
    </tr>
    <tr>
      <td rowspan="2">Multi-objective Optimization</td>
      <td><b>Dominance-based</b></td>
      <td>NSGA-II, NSGA-III</td>
    </tr>
    <tr>
      <td><b>Decomposition / Indicator-based</b></td>
      <td>MOEA/D, RVEA, RVEA-a, HypE</td>
    </tr>
    <tr>
      <td rowspan="3">Benchmark Problems</td>
      <td><b>Basic numerical</b></td>
      <td>Sphere, Ackley, Rastrigin, Rosenbrock, Griewank, Schwefel, Ellipsoid, Levy, Zakharov, ShiftAffineNumericalProblem</td>
    </tr>
    <tr>
      <td><b>Multi-objective numerical</b></td>
      <td>DTLZ1–DTLZ7 (with Pareto-front helpers), CEC2022 (all 12 functions)</td>
    </tr>
    <tr>
      <td><b>Virtual / neuroevolution</b></td>
      <td>VirtualProblem, VirtualLoRAProblem (Gaussian-noise virtual policies)</td>
    </tr>
    <tr>
      <td rowspan="3">Tooling</td>
      <td><b>Operators</b></td>
      <td>15 operators: selection, crossover, mutation, sampling</td>
    </tr>
    <tr>
      <td><b>Metrics</b></td>
      <td>GD, IGD, HV (+ `gd_plus`, `igd_plus`)</td>
    </tr>
    <tr>
      <td><b>Workflow / HPO / Extensions</b></td>
      <td>StdWorkflow + EvalMonitor, host-side HPO wrapper, `evox_etl_ext` plugin namespace</td>
    </tr>
  </tbody>
</table>

For a comprehensive list and detailed descriptions, check the [Tutorial](https://evox.readthedocs.io/en/latest/tutorial/index.html),
the [API Reference](https://evox.readthedocs.io/en/latest/apidocs/index.html), and
[`src/evox_etl/DESIGN.md`](src/evox_etl/DESIGN.md) for the binding design specification.

## Installation

**EvoX 2.0 requires Python ≥ 3.11.** Its only runtime dependencies are `evoxc >= 0.1.0` and `numpy`; every compiled
backend is an opt-in extra, so the default install runs the pure-NumPy backend immediately.

### Which version should I install?

The distribution name is `evox` for both lines. PyPI does not pick a pre-release unless you ask for one, so nothing
changes for users on the stable line:

```bash
# stable 1.x line — unaffected by these pre-releases
pip install "evox[default]"
```

To try the 2.0 pre-release without disturbing that, use a separate virtual environment:

```bash
python -m venv .venv-evox2 && . .venv-evox2/bin/activate   # Windows: .venv-evox2\Scripts\activate
pip install --pre evox==2.0.0a1
# or, tracking every 2.0 pre-release from this point on:
pip install "evox>=2.0.0a1"
```

Pin the exact version (`evox==2.0.0a1`) if you need reproducibility: 2.0 is a pre-release line and its API may
change between pre-releases.

### Backends and optional extras

| Extra | Installs | Enables |
|---|---|---|
| *(none)* | `evoxc`, `numpy` | `backend="numpy"` — the default pure-NumPy interpreter, CPU only, no toolchain |
| `evox[iree]` | `evoxc[iree]` | `backend="iree"` — compiled CPU, NVIDIA CUDA and AMD ROCm |
| `evox[tvm]` | `evoxc[tvm]` | `backend="tvm"` — compiled CPU |
| `evox[compilers]` | both compiled toolchains | `"iree"` and `"tvm"` in one environment |
| `evox[vis]` | `plotly`, `pandas` | `EvalMonitor.plot()`, `evox_etl.vis_tools` figures and `.exv` files |
| `evox[all]` | both compiled toolchains + `plotly`/`pandas` | everything above |

```bash
pip install --pre "evox[vis]"          # plotting on top of the pre-release
pip install --pre "evox[iree]"         # IREE: compiled CPU / CUDA / ROCm
pip install --pre "evox[compilers]"    # IREE + TVM
```

The compiled backends come from the compiler's own extras; `evoxc[iree]` / `evoxc[tvm]` / `evoxc[compilers]` are
the equivalent spellings when you manage `evoxc` directly. The **XLA** backend has no pip extra: it loads a PJRT
plugin that you supply, and `evoxc` finds it through the `EVOXC_PJRT_PLUGIN` environment variable, which must be
exported before the process starts:

```bash
export EVOXC_PJRT_PLUGIN=/path/to/pjrt_c_api_cpu_plugin.so
```

### From source

```bash
git clone https://github.com/EMI-Group/evox.git
cd evox
pip install -e .
```

`src/evox_etl` and `src/evox_etl_ext` are PEP 420 namespace packages, so uninstalled checkouts need
`PYTHONPATH=src` (the runnable scripts under `examples/` add it for you). Contributor setup, the test venv and the
suite layout are documented in the [Developer Guide](https://evox.readthedocs.io/en/latest/guide/developer/index.html).

## Quick Start

A single-objective run — PSO on the 10-D Sphere function — uses the default NumPy backend and needs nothing but
`evoxc` and NumPy:

```python
from evox_etl import EvalMonitorConfig, StdWorkflow
from evox_etl.algorithms import make_pso
from evox_etl.problems import Sphere

dim = 10
workflow = StdWorkflow(
    algorithm=make_pso(100, lb=[-100.0] * dim, ub=[100.0] * dim),
    problem=Sphere(),
    monitor=EvalMonitorConfig(),
    opt_direction="min",
    num_generations=50,
    backend="numpy",  # "iree" / "tvm" / "xla" compile the very same program
)
workflow.run(50, seed=42)                     # init + 50 generations

print(workflow.monitor.get_best_fitness())    # optimum of Sphere is 0.0
workflow.monitor.plot()                       # Plotly figure (needs `evox[vis]`)
```

The run prints the best fitness found after 50 generations (the Sphere optimum is `0.0`):

```console
0.027914267033338547
```

A multi-objective run — NSGA-II on 2-objective DTLZ2 — driven generation by generation, which is the same loop
`run()` performs:

```python
import numpy as np
from evox_etl import EvalMonitorConfig, StdWorkflow
from evox_etl.algorithms import make_nsga2
from evox_etl.problems import DTLZ2

dim = 5
workflow = StdWorkflow(
    algorithm=make_nsga2(pop_size=40, n_objs=2, lb=[0.0] * dim, ub=[1.0] * dim),
    problem=DTLZ2(d=dim, m=2),
    monitor=EvalMonitorConfig(),
    opt_direction=["min", "min"],   # one direction per objective
    num_generations=50,
)

state = workflow.init(seed=42)
state = workflow.init_step(state)
for _ in range(49):
    state = workflow.step(state)
workflow.flush()                                 # hand the last generation to the monitor

pareto = workflow.monitor.get_pf_fitness()       # (n_pareto, n_objs), un-negated
# the DTLZ2 Pareto front is the unit sphere, so |f| == 1 for every optimal point
print(pareto.shape, np.abs(np.linalg.norm(pareto, axis=1) - 1.0).max())
```

It prints the size of the approximate Pareto front, then the largest deviation of its points from the true unit
sphere:

```console
(350, 2) 0.005379677
```

> [!NOTE]
> For fuller walkthroughs see [`examples/quickstart.py`](examples/quickstart.py) (verified to run) and the
> [tutorial series](https://evox.readthedocs.io/en/latest/tutorial/index.html), which starts from a clean
> environment and ends with multi-objective metrics and `.exv` export.

## Programming Model

EvoX 2.0 is functional end to end, in three steps:

1. **Describe the components with frozen configs.** `make_pso(...)`, `make_nsga2(...)`, `Sphere()`,
   `DTLZ2(d=5, m=2)`, `EvalMonitorConfig()` all return dataclasses holding static hyperparameters (Python scalars
   and flat float tuples). Configs are never mutated and never traced.
2. **Let `StdWorkflow` own the loop.** `StdWorkflow(algorithm=..., problem=..., monitor=..., opt_direction=...,
   num_generations=..., backend=...)` builds one step graph through `evoxc` and runs it in a same-device loop:
   `wf.run(n, seed=42)` for the whole run, or `wf.init(seed=...)` → `wf.init_step(state)` → `wf.step(state)` →
   `wf.final_step(state)` when you want to drive it yourself.
3. **Read results from the monitor.** The workflow keeps a host-side `EvalMonitor` at `wf.monitor`
   (`get_best_fitness()`, `get_best_solution()`, `get_fitness_history()`, `get_pf_fitness()`, `plot()`),
   so nothing has to be pulled out of a compiled graph by hand.

The component contract is equally plain: an algorithm module exposes `init(config, key) -> state` and
`step(config, state, evaluate) -> state` (optionally `init_step`/`final_step`), a problem exposes
`evaluate(config, state, candidates)`, and a monitor exposes `init`/`step`/`record_*`. Randomness is a stateless key
— `evox_etl.key(seed)` and `evox_etl.random.*` — stored in the state and advanced there, so runs are reproducible
without any global seed. Internally everything minimizes; `opt_direction="max"` (or a per-objective list) makes the
workflow flip signs. Only algorithms, problems and the step graph live in a trace: monitors, metrics and HPO stay
host-side.

## What Changed Since 1.x

EvoX 2.0 is a rewrite, not a wrapper: `import evox_etl` touches neither the 1.x tree nor PyTorch. The 1.x
object-oriented API — `import evox`, `ModuleBase`, `Parameter`/`Mutable`, `evox.core.compile`/`vmap`/`use_state`,
class-based components, the `enable_distributed` multi-rank switch and the Triton kernel package — **has been
removed**, together with the `src/evox/` tree; there is no deprecation shim.

| Area | 1.x (removed) | 2.0 (`evox_etl`) |
|---|---|---|
| Import | `evox` | `evox_etl` |
| Components | `torch.nn.Module` subclasses, mutable attributes | frozen config dataclasses + `make_*` constructors |
| Execution | `wf.init_step()` / `wf.step()`, mutating internals | `state = wf.init(seed=...)` → `wf.step(state)`, or `wf.run(n, seed=...)` |
| Randomness | `torch.manual_seed(...)` | stateless keys in the state, seeded by `seed=` |
| Compilation | `torch.compile` / `vmap` on user code | one step graph built by `StdWorkflow` through `evoxc` |
| Backends | PyTorch device placement | `backend="numpy"/"iree"/"tvm"/"xla"` + `device=` |
| Metrics | `hv(objs, ref)` | `hv(key, objs, ref, num_sample=...)` — hypervolume takes a leading RNG key, and `gd`/`igd` gain `_plus` variants |
| Pareto front | `DTLZ2(...).pf()` method | module-level `pf(config)` next to each DTLZ definition |

**Not in this pre-release yet** (each was deliberately dropped rather than half-ported):

- **No distributed multi-rank workflow.** Every workflow is single-device; run one process per GPU or scale with
  your own launcher.
- **No Brax / MuJoCo Playground / supervised-learning problems.** Only the virtual (Gaussian-noise neuroevolution)
  problems are ported, so there is no reinforcement-learning environment integration yet.
- **No `ParamsAndVector` / `torch.nn` interop.** Virtual problems take pre-flattened parameter shapes and layer
  specs instead of a `torch.nn.Module`.
- **No Triton kernels.** The hand-written CUDA kernels had no compiler analogue; the virtual-population noise they
  accelerated is regenerated in-graph by a deterministic splitmix64 + Box-Muller implementation.
- **No `torch.compile` / `vmap` internals.** There is nothing to decorate: the workflow compiles one graph with
  `evoxc.build` and runs it, and operators are already batched tensor functions.
- **HPO is host-side, not a `Problem`.** `HPOProblemWrapper` runs a fresh inner `StdWorkflow` per candidate on the
  host instead of nesting a compiled workflow inside an outer trace (which the compiler cannot do); drive it with
  `random_search` or any external optimizer. See [`examples/hpo/`](examples/hpo) for working scripts.

If you are porting 1.x code, read the [migration guide](https://evox.readthedocs.io/en/latest/tutorial/tutorial_part7.html):
it maps every removed API to its replacement.

## Documentation

- **Installation guide** — [English](https://evox.readthedocs.io/en/latest/guide/install/index.html) · [中文](https://evox.readthedocs.io/zh-cn/latest/index.html)
- **Tutorial (7 parts)** — [English](https://evox.readthedocs.io/en/latest/tutorial/index.html) · [中文](https://evox.readthedocs.io/zh-cn/latest/index.html)
- **Developer guide** (architecture, custom algorithms/problems/monitors, contributor setup) — [English](https://evox.readthedocs.io/en/latest/guide/developer/index.html)
- **Advanced guide** (backends, operators, metrics, visualization/`.exv`, HPO, extensions) — [English](https://evox.readthedocs.io/en/latest/guide/advanced/index.html)
- **API reference** — [English](https://evox.readthedocs.io/en/latest/apidocs/index.html)
- **Examples** — [English](https://evox.readthedocs.io/en/latest/examples/index.html), plus runnable scripts in
  [`examples/`](examples) (`quickstart.py`, `examples/hpo/`)
- **Design and history** — [`src/evox_etl/DESIGN.md`](src/evox_etl/DESIGN.md) (binding spec),
  [`REFACTOR_REPORT.md`](REFACTOR_REPORT.md) (1.4 → 2.0 refactor report),
  [`benchmarks/etl_vs_torch/`](benchmarks/etl_vs_torch) (backend comparison harness and results)

## Sister Projects
- **EvoRL**: GPU-accelerated framework for Evolutionary Reinforcement Learning. Check out [here](https://github.com/EMI-Group/evorl).
- **EvoGP**: GPU-accelerated framework for Genetic Programming. Check out [here](https://github.com/EMI-Group/evogp).
- **EvoMO**: GPU-accelerated library for Evolutionary Multiobjective Optimization (EMO). Check out [here](https://github.com/EMI-Group/evomo).
- **TensorNEAT**: Tensorized NeuroEvolution of Augmenting Topologies (NEAT) for GPU Acceleration. Check out [here](https://github.com/EMI-Group/tensorneat).
- **TensorRVEA**: Tensorized Reference Vector Guided Evolutionary Algorithm (RVEA) for GPU Acceleration. Check out [here](https://github.com/EMI-Group/tensorrvea).
- **TensorACO**: Tensorized Ant Colony Optimization (ACO) for GPU Acceleration. Check out [here](https://github.com/EMI-Group/tensoraco).
- **EvoXBench**: A real-world benchmark platform for solving various optimization problems, such as Neural Architecture Search (NAS). It operates without the need for GPUs/PyTorch/TensorFlow and supports multiple programming environments. Check out [here](https://github.com/EMI-Group/evoxbench).

Together with EvoX, these and the other EvoX-related libraries published by the [EMI-Group](https://github.com/EMI-Group) form an ecosystem providing **140+ algorithms** and **240+ benchmark problems/environments**.

Stay tuned - more exciting developments are on the way!  ✨

## Community & Support

- Join discussions on the [GitHub Discussion Board](https://github.com/EMI-Group/evox/discussions).
- Connect via [Discord](https://discord.gg/Vbtgcpy7G4) or QQ group (ID: 297969717).
- Visit [Offical Website](https://evox.group/).

## Citing EvoX

If EvoX contributes to your research, please cite it:

```bibtex
@article{evox,
  title = {{EvoX}: {A} {Distributed} {GPU}-accelerated {Framework} for {Scalable} {Evolutionary} {Computation}},
  author = {Huang, Beichen and Cheng, Ran and Li, Zhuozhao and Jin, Yaochu and Tan, Kay Chen},
  journal = {IEEE Transactions on Evolutionary Computation},
  year = 2024,
  doi = {10.1109/TEVC.2024.3388550}
}
```

## License Notice

EvoX is licensed under the **GNU General Public License v3.0 (GPL-3.0)**. For full terms and conditions, please refer to the [LICENSE](./LICENSE) file.

<!--
## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=EMI-Group/evox&type=Date)](https://star-history.com/#EMI-Group/evox&Date)
-->
