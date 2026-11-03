<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/source/_static/evox_brand_light.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/source/_static/evox_brand_dark.svg">
    <img src="docs/source/_static/evox_brand_dark.svg" alt="EvoX" width="280">
  </picture>
</p>

<h1 align="center">面向演化式 AI 系统的<br>基础设施</h1>

<p align="center">让智能跨越数字与物理世界，持续适应、创造与演化。</p>

<p align="center">
  <a href="https://pypi.org/project/evox/"><img src="https://img.shields.io/pypi/v/evox?color=c8383c" alt="PyPI 稳定版"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.6%2B-c8383c" alt="PyTorch 2.6 及以上"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-555" alt="Python 3.10 及以上"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-GPL--3.0--or--later-555" alt="GPL-3.0-or-later"></a>
</p>

<p align="center">
  <a href="https://www.evox.group/zh-cn/">官网</a> ·
  <a href="https://evox.readthedocs.io/zh-cn/latest/">文档</a> ·
  <a href="https://github.com/EMI-Group/evox/releases">版本发布</a> ·
  <a href="./README.md">English</a>
</p>

**生成候选，检验效果，保留有效方案并继续改进。** EvoX 将这一演化过程带入 AI，从模型参数与控制策略，到科学与工程设计。EvoX 生态连接演化算法、应用库与智能系统，让能力在持续评估中迭代。

**本仓库提供基于 PyTorch 的演化计算框架。** 将候选解及其评估组织为 CPU、GPU 上的批量张量计算，组合算法、问题与监测器即可运行搜索，再用评估结果推动下一代。

<picture>
  <source media="(prefers-reduced-motion: reduce)" srcset="docs/source/_static/readme-compute-zh.svg">
  <img src="docs/source/_static/readme-compute-zh.gif" width="960" alt="候选解组成批量张量并行评估，评估结果反馈到搜索，保留较优候选并通过变化形成下一代。">
</picture>

[查看静态图](docs/source/_static/readme-compute-zh.svg)

## 选择版本

| | **EvoX 1.x · 主版本** | **EvoX 2.0 alpha · 实验版** |
|---|---|---|
| 分支 | [`main`](https://github.com/EMI-Group/evox/tree/main) | [`evoxc-exp`](https://github.com/EMI-Group/evox/tree/evoxc-exp) |
| 计算基础 | **PyTorch** | **[EvoX Compiler（`evoxc`）](https://github.com/EMI-Group/evoxc)** |
| Python 导入 | `import evox` | `import evox_etl` |
| Python 版本 | 3.10+ | 3.11+ |
| 适合用途 | 应用开发、科研实验、PyTorch 集成 | 探索新的函数式 API 与编译后端 |

**首次使用，建议从 [1.x 快速开始](#快速开始)入手。** 实验版基于 EvoX Compiler 进行函数式重写，提供 NumPy 解释执行及可选的 IREE、XLA、TVM 后端，API 仍在演进。两个版本共用 `evox` 包名；试用 alpha 版时，请在独立环境中按照其[安装说明](https://github.com/EMI-Group/evox/blob/evoxc-exp/README.md#installation)操作。

早期 JAX 实现保留在 [`v0.9.0`](https://github.com/EMI-Group/evox/tree/v0.9.0)。

## EvoX 生态

通过生成与评估持续改进，这一思想延伸到四个应用方向。以下相关项目有各自的安装包与运行环境。

| **生成式 AI** | **智能体 AI** |
|---|---|
| <picture><source media="(prefers-reduced-motion: reduce)" srcset="docs/source/_static/readme-generative.png"><img src="docs/source/_static/readme-generative.gif" width="440" alt="通过评估迭代专家模型的融合方案。"></picture> | <picture><source media="(prefers-reduced-motion: reduce)" srcset="docs/source/_static/readme-agentic.png"><img src="docs/source/_static/readme-agentic.gif" width="440" alt="从任务生成源代码，运行测试，并修复测试发现的问题。"></picture> |
| **演化模型。**<br>[EvoGM](https://github.com/JiangTao97/evogm) · 演化式模型融合 | **演化软件。**<br>[EvoX Genesis](https://github.com/EMI-Group/genesis) · 递归智能体系统 |

| **物理 AI** | **AI for Science** |
|---|---|
| <picture><source media="(prefers-reduced-motion: reduce)" srcset="docs/source/_static/readme-physical.png"><img src="docs/source/_static/readme-physical.gif" width="440" alt="在仿真中评估机器人控制策略。"></picture> | <picture><source media="(prefers-reduced-motion: reduce)" srcset="docs/source/_static/readme-science.png"><img src="docs/source/_static/readme-science.gif" width="440" alt="通过评估逐步改进候选翼型设计。"></picture> |
| **演化行为。**<br>[EvoRL](https://github.com/EMI-Group/evorl) · 演化强化学习 | **演化设计。**<br>[EvoGO](https://github.com/EMI-Group/evogo) · 生成式优化 |

专用库还包括面向多目标优化的 [EvoMO](https://github.com/EMI-Group/evomo)、遗传编程的 [EvoGP](https://github.com/EMI-Group/evogp)、拓扑演化的 [TensorNEAT](https://github.com/EMI-Group/tensorneat)，以及神经架构搜索基准平台 [EvoXBench](https://github.com/EMI-Group/evoxbench)。

浏览[完整生态](https://www.evox.group/zh-cn/libs)，或了解实验版使用的 [EvoX Compiler](https://github.com/EMI-Group/evoxc)。

## 快速开始

先安装[适合硬件的 PyTorch](https://pytorch.org/get-started/locally/)，再安装包含可视化功能的 EvoX 1.x：

```bash
python -m pip install "evox[default]>=1.4,<2"
```

### 运行第一次搜索

用粒子群优化算法求解 10 维 Ackley 函数：

```python
import torch
from evox.algorithms import PSO
from evox.problems.numerical import Ackley
from evox.workflows import EvalMonitor, StdWorkflow

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

algorithm = PSO(
    pop_size=100,
    lb=-32 * torch.ones(10),
    ub=32 * torch.ones(10),
)
monitor = EvalMonitor()
workflow = StdWorkflow(algorithm, Ackley(), monitor)

workflow.init_step()
for _ in range(100):
    workflow.step()

print("Best fitness:", monitor.get_best_fitness().item())
# monitor.plot().show()  # Optional: inspect the convergence curve.
```

**Algorithm** 生成并更新候选解，**Problem** 批量评估，**Workflow** 组织运行，**Monitor** 记录结果。

<details>
<summary><strong>使用自己的目标函数</strong></summary>

实现一个批量评估函数。单目标问题接收形状为 `(population_size, dimensions)` 的张量，为每个候选解返回一个适应度值：

```python
from evox.core import Problem

class Sphere(Problem):
    def evaluate(self, candidates: torch.Tensor) -> torch.Tensor:
        return candidates.square().sum(dim=-1)
```

构建工作流时，将 `Ackley()` 替换为 `Sphere()` 即可。EvoX 默认最小化目标；最大化问题可在 `StdWorkflow` 中设置 `opt_direction="max"`。

</details>

<details>
<summary><strong>优化多个目标</strong></summary>

同一套工作流也支持多目标搜索。下面用 RVEA 求解 DTLZ2：

```python
import torch
from evox.algorithms import RVEA
from evox.problems.numerical import DTLZ2
from evox.workflows import EvalMonitor, StdWorkflow

torch.manual_seed(42)
algorithm = RVEA(
    pop_size=100,
    n_objs=2,
    lb=torch.zeros(12),
    ub=torch.ones(12),
)
monitor = EvalMonitor(multi_obj=True)
workflow = StdWorkflow(algorithm, DTLZ2(m=2), monitor)

workflow.init_step()
for _ in range(100):
    workflow.step()

print("Pareto-front fitness shape:", monitor.get_pf_fitness().shape)
# monitor.plot().show()
```

</details>

接下来可阅读[系列教程](https://evox.readthedocs.io/zh-cn/latest/tutorial/index.html)、[神经演化示例](docs/source/examples/brax.ipynb)或[超参数优化示例](docs/source/examples/hpo.ipynb)。

## 为批量搜索而设计

- **张量化算法。** 种群更新与适应度评估按批执行，配合 `evox.compile` 和 `evox.vmap` 进行编译与向量化。
- **可组合的组件。** 更换算法、目标函数或监测器时，复用同一优化流程；使用 PyTorch 编写自定义组件。
- **完整的实验工具。** 跟踪收敛、查看 Pareto 前沿、计算多目标指标，并以 `.exv` 格式导出运行数据。

主包内置 **30+ 种算法**。部分组件如下：

| 任务 | 算法与工具 |
|---|---|
| 单目标优化 | PSO、DE、SHADE、CMA-ES、OpenES、xNES |
| 多目标优化 | NSGA-II、NSGA-III、MOEA/D、RVEA、HypE |
| 数值基准测试 | Ackley、Sphere、Rosenbrock、DTLZ、CEC2022 |
| 神经演化 | 神经网络参数搜索；Brax、MuJoCo Playground 与监督学习问题适配器 |
| 超参数优化 | 嵌套工作流与 `HPOProblemWrapper` |

完整列表见[算法源码](src/evox/algorithms)与[问题源码](src/evox/problems)。神经演化和仿真集成需要额外安装 `neuroevolution` 依赖：

```bash
python -m pip install "evox[default,neuroevolution]>=1.4,<2"
```

### 执行与硬件

PyTorch 版本通过对应后端运行于 **CPU**、**NVIDIA GPU（CUDA）**、**AMD GPU（ROCm）**及 **Apple Silicon（MPS，eager 执行）**。配置方式见[硬件指南](docs/source/miscellaneous/other_gpus.md)。

在支持编译的环境中，可在进入搜索循环前编译反复执行的步骤：

```python
from evox import compile

workflow.step = compile(workflow.step)
```

[多 GPU 与分布式工作流](docs/source/guide/experimental/distributed_workflow.md)目前为实验功能，需要配置分布式环境并使用相应方式启动。可复现的性能实验见[基准测试脚本](benchmarks)。

## 社区与贡献

- **提问与讨论：** [GitHub Discussions](https://github.com/EMI-Group/evox/discussions)、[Discord](https://discord.gg/Vbtgcpy7G4) 或 [QQ 群 297969717](https://qm.qq.com/q/vTPvoMUGAw)。
- **问题反馈与代码贡献：** [提交 Issue](https://github.com/EMI-Group/evox/issues)，或从[开发者指南](https://evox.readthedocs.io/zh-cn/latest/guide/developer/index.html)开始。
- **研究合作：** [程然主页](https://chengran.tech/)。

<details>
<summary>从源码安装主版本</summary>

```bash
git clone --branch main https://github.com/EMI-Group/evox.git
cd evox
python -m pip install -e ".[default]"
```

</details>

## 引用

如果 EvoX 对你的研究有帮助，请引用[框架论文](https://arxiv.org/abs/2301.12457)：

```bibtex
@article{evox,
  title = {{EvoX}: A Distributed GPU-accelerated Framework for Scalable Evolutionary Computation},
  author = {Huang, Beichen and Cheng, Ran and Li, Zhuozhao and Jin, Yaochu and Tan, Kay Chen},
  journal = {IEEE Transactions on Evolutionary Computation},
  year = {2024},
  doi = {10.1109/TEVC.2024.3388550}
}
```

## 许可证

EvoX 使用 [GPL-3.0-or-later](LICENSE) 许可证。EvoX Compiler 等相关项目采用各自的许可证。
