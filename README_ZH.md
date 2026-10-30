<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/source/_static/evox_brand_light.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/source/_static/evox_brand_dark.svg">
    <img alt="EvoX Logo" height="128" width="500px" src="docs/source/_static/evox_brand_dark.svg">
  </picture>
</h1>

## ⚠️ 实验性预发布版本（2.0.0a1）

> [!WARNING]
> 本分支是 **EvoX 2.0 实验性预发布版本（`2.0.0a1`）**：基于 [**EvoX Compiler**](https://github.com/EMI-Group/evoxc)
> （`evoxc`）张量库，从零重写的**函数式** EvoX。
> 它**不是** 1.x 的 PyTorch API：包导入名为 **`evox_etl`**（2.0.0a1 中不存在 `import evox` 模块），1.x 的 PyTorch
> 实现已从本分支中删除。
> 发布名依然是 **`evox`**，因此 PyPI 将 `2.0.0a1` 视为**预发布版本**：直接 `pip install evox` 仍会得到稳定的
> **1.x** 版本，这也是大多数用户推荐的选择。
> 只有 `pip install --pre evox==2.0.0a1`（或 `pip install 'evox>=2.0.0a1'`）才会安装新版本线 —— 请在**独立的虚拟环境**
> 中安装，因为两条版本线共用 `evox` 这一发布名，无法共存于同一环境。
> 在预发布之间，API 仍可能变化。

<div align="center">
  <a href="https://arxiv.org/abs/2301.12457"><img src="https://img.shields.io/badge/arxiv-2212.05652-red" alt="arXiv"></a>
  <a href="https://evox.readthedocs.io/en/latest/index.html"><img src="https://img.shields.io/badge/readthedocs-docs-green?logo=readthedocs" alt="文档"></a>
  <a href="https://pypi.org/project/evox/"><img src="https://img.shields.io/pypi/v/evox?logo=python" alt="PyPI 版本"></a>
  <a href="https://pypi.org/project/evox/"><img src="https://img.shields.io/badge/python-3.11+-orange?logo=python" alt="Python 版本"></a>
  <a href="https://discord.gg/Vbtgcpy7G4"><img src="https://img.shields.io/badge/discord-evox-%235865f2?logo=discord" alt="Discord 服务器"></a>
  <a href="https://qm.qq.com/q/vTPvoMUGAw"><img src="https://img.shields.io/badge/QQ-297969717-%231db4f4?logo=tencentqq" alt="QQ 群"></a>
</div>

<p align="center">
  <a href="./README.md"><img src="https://img.shields.io/badge/English-f6f5f4" alt="English README"></a>
  <a href="./README_ZH.md"><img src="https://img.shields.io/badge/中文-f6f5f4" alt="中文 README"></a>
</p>

---

<h3 align="center"> 🌟函数式、编译器驱动的大规模演化计算框架🌟 </h3>

---

## 🔥 新闻
- [2026-10-30] 发布 **EvoX 2.0.0a1**（实验性预发布）—— EvoX 现已基于 **EvoX Compiler**（`evoxc`）从零重写为**函数式**框架，导入名为 `evox_etl`。[[发布说明](https://github.com/EMI-Group/evox/releases/tag/v2.0.0a1)]
- [2026-09-09] 发布 **EvoX 1.4.0** - 🚀 维护版本更新，包含多项修复与改进！ [[发布说明](https://github.com/EMI-Group/evox/releases/tag/v1.4.0)]

## 目录

1. [实验性预发布版本（2.0.0a1）](#-实验性预发布版本2000a1)
2. [概述](#概述)
3. [主要特性](#主要特性)
4. [主要内容](#主要内容)
5. [安装](#安装)
6. [快速开始](#快速开始)
7. [编程模型](#编程模型)
8. [与 1.x 相比的变化](#与-1x-相比的变化)
9. [文档](#文档)
10. [相关项目](#相关项目)
11. [社区支持](#社区支持)
12. [引用 EvoX](#引用-evox)

## 概述

EvoX 是一个面向可扩展演化计算的 GPU 加速框架。它提供易用的编程模型，EvoX 生态包含 **140+ 进化算法 (EAs)** 和
**240+ 基准问题/环境** —— 其中本包内置 **30+ 种算法** 和 **30+ 基准问题**，其余来自姊妹项目。详情请参阅我们的
[论文](https://arxiv.org/abs/2301.12457) 及 [文档](https://evox.readthedocs.io/zh-cn/latest/index.html) /
[Documentation](https://evox.readthedocs.io/en/latest/index.html)。

EvoX 2.0 保留了这一整套算法与问题，但更换了底层机制：每个算法、问题、监视器与指标都是纯 Python ——
冻结的配置数据类、张量形式的状态，以及 `init`/`step` 函数 —— 组件模型中唯一的编译入口是 `StdWorkflow`，它通过 `evoxc`
追踪一次单步计算图，然后在循环中执行。因此同一个程序只需改动一个参数，就能运行在纯 NumPy 解释器上，或运行在
IREE / TVM / XLA 编译后端的 CPU 与 GPU 上。

> [!NOTE]
> 需要稳定版本？`pip install evox` 目前仍解析到 **1.x** 的 PyTorch 版本线；其更早的 JAX 实现位于 **v0.9.0**
> 分支。本仓库的默认分支即为 2.0 预发布版本。

## 主要特性

### 🚀 编译执行，而非解释执行

- **一个程序，多种执行模式**：`backend="numpy"`（默认，无需任何工具链）、`"iree"`（编译后的 CPU / CUDA / ROCm）、
  `"tvm"`（编译后的 CPU）或 `"xla"`（PJRT）。只需修改 `StdWorkflow(backend=..., device=...)`，算法与问题配置完全不变。
- **一次编译，设备端循环**：`StdWorkflow` 只构建一次单步计算图，之后把输出作为输入回灌，而不是每代重新追踪。
- **GPU 级吞吐**：在仓库自带的 `benchmarks/etl_vs_torch/` 测试中，编译后的 XLA-CUDA 后端在相同的 PSO/DE 用例上
  每步比 1.x 的 torch 实现**快 1.16–1.45 倍**（RTX A6000），且收敛结果完全一致
  （[BENCHMARK_RESULTS.md](benchmarks/etl_vs_torch/BENCHMARK_RESULTS.md)）。

### 🌐 一体化解决方案

- 内置 **32 个互不相同的算法类**（27 个单目标导出名，其中 `VirtualLoRAES` 是 `VirtualES` 的别名，另加 6 个多目标导出名），
  涵盖 DE、ES、PSO、免训练的 virtual ES 家族、NSGA-II/III、MOEA/D、RVEA、RVEA-a 与 HypE；EvoX 生态更提供
  **140+ 种算法**。
- **30+ 个基准问题**：10 个基础数值问题（9 个经典函数外加一个 shift/affine 包装器）、DTLZ1–7、CEC2022 的全部 12 个函数，
  以及虚拟（高斯噪声神经演化）问题；生态内共有 **240+ 基准问题/环境**。
- **宿主端超参数优化**：通过 `evox_etl.problems.hpo_wrapper` 提供（为何是宿主端实现，见
  [与 1.x 相比的变化](#与-1x-相比的变化)）。

### 🛠️ 易用的函数式设计

- **配置即数据**：每个组件都是一个冻结数据类，通常由 `make_*` 构造函数创建
  （`make_pso(...)`、`make_cma_es(...)`、`make_nsga2(...)`）。
- **状态即数据**：`init(config, key)` 返回由张量组成的 pytree，`step(config, state, evaluate)` 返回下一个状态 ——
  可序列化、可比对、可回放。
- **显式随机性**：无状态的随机键（`evox_etl.key(seed)`）保存在状态中，因此一个整数种子即可复现整次运行；
  不存在全局种子。
- **处处皆普通函数**：算法、算子与指标都是可测试、可复用的普通 Python —— 没有需要继承的框架基类。

### 📊 灵活的可视化与指标

- **开箱即用的分析工具**：`EvalMonitor` 记录适应度、精英个体与种群历史，以及自定义的辅助通道；
  `EvalMonitor.plot()` 返回可交互的 Plotly 图。
- **多目标指标**：`evox_etl.metrics` 提供 `gd`、`igd`（含 `gd_plus`/`igd_plus`）与 `hv`。
- **实时数据流**：定制的 **`.exv`** 格式（`evox_etl.vis_tools`）可序列化整次运行，支持快速增量读取。

## 主要内容

<table border="1" cellspacing="0" cellpadding="8" style="border-collapse: collapse; width: 100%; text-align: left;">
  <thead>
    <tr style="background-color: #f2f2f2;">
      <th>类别</th>
      <th>子类别</th>
      <th>本包内置的算法 / 基准问题</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3">单目标优化</td>
      <td><b>差分演化</b></td>
      <td>DE, CoDE, JaDE, SaDE, SHADE, ODE</td>
    </tr>
    <tr>
      <td><b>演化策略</b></td>
      <td>CMA-ES, OpenES, xNES, SNES, SeparableNES, ARS, ASEBO, DES, ESMC, GuidedES, NoiseReuseES, PersistentES, VirtualES / VirtualLoRAES</td>
    </tr>
    <tr>
      <td><b>粒子群优化</b></td>
      <td>PSO, CLPSO, CSO, DMS-PSO-EL, FS-PSO, SL-PSO-GS, SL-PSO-US</td>
    </tr>
    <tr>
      <td rowspan="2">多目标优化</td>
      <td><b>基于支配关系</b></td>
      <td>NSGA-II, NSGA-III</td>
    </tr>
    <tr>
      <td><b>基于分解 / 指标</b></td>
      <td>MOEA/D, RVEA, RVEA-a, HypE</td>
    </tr>
    <tr>
      <td rowspan="3">基准问题</td>
      <td><b>基础数值问题</b></td>
      <td>Sphere, Ackley, Rastrigin, Rosenbrock, Griewank, Schwefel, Ellipsoid, Levy, Zakharov, ShiftAffineNumericalProblem</td>
    </tr>
    <tr>
      <td><b>多目标数值问题</b></td>
      <td>DTLZ1–DTLZ7（含 Pareto 前沿辅助函数）, CEC2022（全部 12 个函数）</td>
    </tr>
    <tr>
      <td><b>虚拟 / 神经演化</b></td>
      <td>VirtualProblem, VirtualLoRAProblem（高斯噪声虚拟策略）</td>
    </tr>
    <tr>
      <td rowspan="3">工具</td>
      <td><b>算子</b></td>
      <td>15 个算子：选择、交叉、变异、采样</td>
    </tr>
    <tr>
      <td><b>指标</b></td>
      <td>GD, IGD, HV（以及 `gd_plus`、`igd_plus`）</td>
    </tr>
    <tr>
      <td><b>工作流 / HPO / 扩展</b></td>
      <td>StdWorkflow + EvalMonitor、宿主端 HPO 包装器、`evox_etl_ext` 插件命名空间</td>
    </tr>
  </tbody>
</table>

完整的列表与详细说明，请参阅 [教程](https://evox.readthedocs.io/zh-cn/latest/tutorial/index.html)、
[API 参考](https://evox.readthedocs.io/zh-cn/latest/apidocs/index.html)，以及
[`src/evox_etl/DESIGN.md`](src/evox_etl/DESIGN.md)（约束性设计规范）。

## 安装

**EvoX 2.0 需要 Python ≥ 3.11。** 运行时依赖只有 `evoxc >= 0.1.0` 与 `numpy`；所有编译后端都是可选 extra，
因此默认安装即可直接运行纯 NumPy 后端。

### 应该安装哪个版本？

两条版本线共用一个发布名 `evox`。除非显式指定，PyPI 不会选择预发布版本，因此稳定版本线的用户不受任何影响：

```bash
# 稳定的 1.x 版本线 —— 不受这些预发布版本影响
pip install "evox[default]"
```

如需在不受影响的前提下试用 2.0 预发布版本，请使用独立的虚拟环境：

```bash
python -m venv .venv-evox2 && . .venv-evox2/bin/activate   # Windows: .venv-evox2\Scripts\activate
pip install --pre evox==2.0.0a1
# 或者，从此跟踪后续所有 2.0 预发布版本：
pip install "evox>=2.0.0a1"
```

若需要可复现，请固定版本（`evox==2.0.0a1`）：2.0 是预发布版本线，其 API 可能在预发布之间变化。

### 后端与可选 extra

| Extra | 安装内容 | 启用能力 |
|---|---|---|
| *（无）* | `evoxc`、`numpy` | `backend="numpy"` —— 默认的纯 NumPy 解释器，仅 CPU，无需工具链 |
| `evox[iree]` | `evoxc[iree]` | `backend="iree"` —— 编译后的 CPU、NVIDIA CUDA 与 AMD ROCm |
| `evox[tvm]` | `evoxc[tvm]` | `backend="tvm"` —— 编译后的 CPU |
| `evox[compilers]` | 两套编译工具链 | 在同一环境中使用 `"iree"` 与 `"tvm"` |
| `evox[vis]` | `plotly`、`pandas` | `EvalMonitor.plot()`、`evox_etl.vis_tools` 绘图与 `.exv` 文件 |
| `evox[all]` | 两套编译工具链 + `plotly`/`pandas` | 以上全部 |

```bash
pip install --pre "evox[vis]"          # 在预发布版本上启用绘图
pip install --pre "evox[iree]"         # IREE：编译后的 CPU / CUDA / ROCm
pip install --pre "evox[compilers]"    # IREE + TVM
```

编译后端来自编译器自身的 extra；若你自行管理 `evoxc`，等价的写法是 `evoxc[iree]` / `evoxc[tvm]` /
`evoxc[compilers]`。**XLA** 后端没有 pip extra：它需要加载由你提供的 PJRT 插件，`evoxc` 通过 `EVOXC_PJRT_PLUGIN`
环境变量定位该插件，且必须在进程启动前导出：

```bash
export EVOXC_PJRT_PLUGIN=/path/to/pjrt_c_api_cpu_plugin.so
```

### 从源码安装

```bash
git clone https://github.com/EMI-Group/evox.git
cd evox
pip install -e .
```

`src/evox_etl` 与 `src/evox_etl_ext` 是 PEP 420 命名空间包，因此未安装的源码目录需要设置 `PYTHONPATH=src`
（`examples/` 下的可运行脚本会自动处理）。贡献者环境、测试虚拟环境与测试套件结构见
[开发者指南](https://evox.readthedocs.io/zh-cn/latest/guide/developer/index.html)。

## 快速开始

下面是一个单目标示例 —— 用 PSO 求解 10 维 Sphere 函数，使用默认的 NumPy 后端，只需要 `evoxc` 与 NumPy：

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
    backend="numpy",  # "iree" / "tvm" / "xla" 编译完全相同的程序
)
workflow.run(50, seed=42)                     # init + 50 代

print(workflow.monitor.get_best_fitness())    # Sphere 的最优值为 0.0
workflow.monitor.plot()                       # Plotly 图（需要 `evox[vis]`）
```

运行后会打印 50 代之后找到的最优适应度（Sphere 的最优值为 `0.0`）：

```console
0.027914267033338547
```

下面是一个多目标示例 —— 用 NSGA-II 求解 2 目标 DTLZ2，并逐代驱动，这正是 `run()` 内部执行的同一个循环：

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
    opt_direction=["min", "min"],   # 每个目标一个方向
    num_generations=50,
)

state = workflow.init(seed=42)
state = workflow.init_step(state)
for _ in range(49):
    state = workflow.step(state)
workflow.flush()                                 # 把最后一代交给监视器

pareto = workflow.monitor.get_pf_fitness()       # (n_pareto, n_objs)，未取反
# DTLZ2 的 Pareto 前沿是单位球面，因此每个最优点都满足 |f| == 1
print(pareto.shape, np.abs(np.linalg.norm(pareto, axis=1) - 1.0).max())
```

它会打印近似 Pareto 前沿的点数，以及这些点到真实单位球面的最大偏差：

```console
(350, 2) 0.005379677
```

> [!NOTE]
> 更完整的示例见 [`examples/quickstart.py`](examples/quickstart.py)（已确认可运行）以及
> [教程系列](https://evox.readthedocs.io/zh-cn/latest/tutorial/index.html)，后者从干净的环境开始，
> 一直到多目标指标与 `.exv` 导出。

## 编程模型

EvoX 2.0 端到端都是函数式的，分三步：

1. **用冻结配置描述组件。** `make_pso(...)`、`make_nsga2(...)`、`Sphere()`、`DTLZ2(d=5, m=2)`、
   `EvalMonitorConfig()` 都返回保存静态超参数（Python 标量与扁平的 float 元组）的数据类。配置永不被修改，
   也永不被追踪。
2. **把循环交给 `StdWorkflow`。** `StdWorkflow(algorithm=..., problem=..., monitor=..., opt_direction=...,
   num_generations=..., backend=...)` 通过 `evoxc` 构建一次单步计算图，并在同一设备上循环执行：
   用 `wf.run(n, seed=42)` 跑完整个流程，或在你希望自行驱动时使用 `wf.init(seed=...)` →
   `wf.init_step(state)` → `wf.step(state)` → `wf.final_step(state)`。
3. **从监视器读取结果。** 工作流在 `wf.monitor` 上维护一个宿主端的 `EvalMonitor`
   （`get_best_fitness()`、`get_best_solution()`、`get_fitness_history()`、`get_pf_fitness()`、`plot()`），
   无需手工从编译后的计算图中取数据。

组件契约同样简单：算法模块提供 `init(config, key) -> state` 与 `step(config, state, evaluate) -> state`
（可选 `init_step`/`final_step`），问题提供 `evaluate(config, state, candidates)`，监视器提供
`init`/`step`/`record_*`。随机性是保存在状态中的无状态键 —— `evox_etl.key(seed)` 与 `evox_etl.random.*` —— 并在
状态中推进，因此无需任何全局种子即可复现运行。框架内部一律做最小化；`opt_direction="max"`（或按目标给出的列表）
让工作流翻转符号。只有算法、问题与单步计算图位于追踪之中：监视器、指标与 HPO 都留在宿主端。

## 与 1.x 相比的变化

EvoX 2.0 是一次重写，而不是一层封装：`import evox_etl` 既不触及 1.x 的代码树，也不依赖 PyTorch。1.x 的面向对象
API —— `import evox`、`ModuleBase`、`Parameter`/`Mutable`、`evox.core.compile`/`vmap`/`use_state`、基于类的组件、
`enable_distributed` 多进程开关以及 Triton 内核包 —— **已被移除**，`src/evox/` 目录也一并删除；不提供任何弃用兼容层。

| 方面 | 1.x（已移除） | 2.0（`evox_etl`） |
|---|---|---|
| 导入 | `evox` | `evox_etl` |
| 组件 | `torch.nn.Module` 子类、可变属性 | 冻结配置数据类 + `make_*` 构造函数 |
| 执行 | `wf.init_step()` / `wf.step()`，修改内部状态 | `state = wf.init(seed=...)` → `wf.step(state)`，或 `wf.run(n, seed=...)` |
| 随机性 | `torch.manual_seed(...)` | 状态中的无状态键，由 `seed=` 初始化 |
| 编译 | 对用户代码使用 `torch.compile` / `vmap` | 由 `StdWorkflow` 通过 `evoxc` 构建一次单步计算图 |
| 后端 | PyTorch 设备放置 | `backend="numpy"/"iree"/"tvm"/"xla"` + `device=` |
| 指标 | `hv(objs, ref)` | `hv(key, objs, ref, num_sample=...)` —— 超体积多一个前置随机键参数，`gd`/`igd` 另有 `_plus` 变体 |
| Pareto 前沿 | `DTLZ2(...).pf()` 方法 | 每个 DTLZ 定义旁的模块级 `pf(config)` |

**本预发布版本尚未包含的功能**（这些功能是有意舍弃，而非半成品移植）：

- **没有分布式多进程工作流。** 每个工作流都是单设备的；请每个 GPU 启动一个进程，或使用你自己的启动器来扩展。
- **没有 Brax / MuJoCo Playground / 监督学习问题。** 只移植了虚拟（高斯噪声神经演化）问题，因此暂时还没有
  强化学习环境集成。
- **没有 `ParamsAndVector` / `torch.nn` 互操作。** 虚拟问题接收预先展平的参数形状与层规格，而不是 `torch.nn.Module`。
- **没有 Triton 内核。** 手写 CUDA 内核在编译器中没有对应物；它们原本加速的虚拟种群噪声改由确定性的
  splitmix64 + Box-Muller 在计算图内生成。
- **没有 `torch.compile` / `vmap` 内部机制。** 没有需要装饰的对象：工作流用 `evoxc.build` 编译一张图并执行它，
  而算子本身已经是批量张量函数。
- **HPO 是宿主端实现，而不是 `Problem`。** `HPOProblemWrapper` 在宿主端为每个候选重新运行一次内部
  `StdWorkflow`，而不是把编译后的工作流嵌套在外层追踪之中（编译器无法做到）；可用 `random_search` 或任意外部
  优化器驱动它。可运行的脚本见 [`examples/hpo/`](examples/hpo)。

如果你正在移植 1.x 代码，请阅读[迁移指南](https://evox.readthedocs.io/zh-cn/latest/tutorial/tutorial_part7.html)：
它把每个被移除的 API 映射到对应的替代方案。

## 文档

- **安装指南** — [中文](https://evox.readthedocs.io/zh-cn/latest/guide/install/index.html) · [English](https://evox.readthedocs.io/en/latest/guide/install/index.html)
- **教程（共 7 部分）** — [中文](https://evox.readthedocs.io/zh-cn/latest/tutorial/index.html) · [English](https://evox.readthedocs.io/en/latest/tutorial/index.html)
- **开发者指南**（架构、自定义算法/问题/监视器、贡献者环境） — [中文](https://evox.readthedocs.io/zh-cn/latest/guide/developer/index.html)
- **进阶指南**（后端、算子、指标、可视化/`.exv`、HPO、扩展） — [中文](https://evox.readthedocs.io/zh-cn/latest/guide/advanced/index.html)
- **API 参考** — [中文](https://evox.readthedocs.io/zh-cn/latest/apidocs/index.html)
- **示例** — [中文](https://evox.readthedocs.io/zh-cn/latest/examples/index.html)，以及 [`examples/`](examples) 下的
  可运行脚本（`quickstart.py`、`examples/hpo/`）
- **设计与历史** — [`src/evox_etl/DESIGN.md`](src/evox_etl/DESIGN.md)（约束性规范）、
  [`REFACTOR_REPORT.md`](REFACTOR_REPORT.md)（1.4 → 2.0 重构报告）、
  [`benchmarks/etl_vs_torch/`](benchmarks/etl_vs_torch)（后端对比工具与结果）

## 相关项目

- **EvoRL**: 基于 GPU 加速的进化强化学习框架。查看详情：[链接](https://github.com/EMI-Group/evorl)。
- **EvoGP**: 基于 GPU 加速的遗传编程框架。查看详情：[链接](https://github.com/EMI-Group/evogp)。
- **EvoMO**: 基于 GPU 加速的进化多目标优化（EMO）库。查看详情：[链接](https://github.com/EMI-Group/evomo)。
- **TensorNEAT**: 用于 GPU 加速的张量化 NEAT（NeuroEvolution of Augmenting Topologies）框架。查看详情：[链接](https://github.com/EMI-Group/tensorneat)。
- **TensorRVEA**: 用于 GPU 加速的张量化参考向量引导进化算法（RVEA）框架。查看详情：[链接](https://github.com/EMI-Group/tensorrvea)。
- **TensorACO**: 用于 GPU 加速的张量化蚁群优化算法（ACO）框架。查看详情：[链接](https://github.com/EMI-Group/tensoraco)。
- **EvoXBench**: 一个用于解决各种优化问题（如神经架构搜索 NAS）的真实世界基准测试平台。该平台无需 GPU/PyTorch/TensorFlow 运行，并支持多种编程环境。查看详情：[链接](https://github.com/EMI-Group/evoxbench)。

EvoX 与这些项目以及其他由 [EMI-Group](https://github.com/EMI-Group) 发布的 EvoX 相关库共同构成了一个提供
**140+ 种算法** 和 **240+ 基准问题/环境** 的生态。

敬请期待——更多精彩内容即将推出！✨

## 社区支持

- 在 [GitHub 讨论区](https://github.com/EMI-Group/evox/discussions) 参与讨论。
- 通过 [Discord](https://discord.gg/Vbtgcpy7G4) 或 QQ 群（ID: 297969717）联系交流。
- 访问 [EvoX官网](https://evox.group/)。

## 引用 EvoX

如果 EvoX 对您的研究有帮助，请引用：

```bibtex
@article{evox,
  title = {{EvoX}: {A} {Distributed} {GPU}-accelerated {Framework} for {Scalable} {Evolutionary} {Computation}},
  author = {Huang, Beichen and Cheng, Ran and Li, Zhuozhao and Jin, Yaochu and Tan, Kay Chen},
  journal = {IEEE Transactions on Evolutionary Computation},
  year = 2024,
  doi = {10.1109/TEVC.2024.3388550}
}
```

## 许可证声明

EvoX 遵循 **GNU 通用公共许可证 3.0 (GPL-3.0)** 进行授权。完整的条款和条件请参阅 [LICENSE](./LICENSE) 文件。

<!--
## Star 历史
[![Star 历史图表](https://api.star-history.com/svg?repos=EMI-Group/evox&type=Date)](https://star-history.com/#EMI-Group/evox&Date)
-->
