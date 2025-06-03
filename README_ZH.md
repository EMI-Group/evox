<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/source/_static/evox_logo_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/source/_static/evox_logo_light.png">
    <img alt="EvoX Logo" height="128" width="500px" src="docs/source/_static/evox_logo_light.png">
  </picture>
</h1>

<p align="center">
  <picture>
    <source type="image/avif" srcset="docs/source/_static/pso_result.avif">
    <img src="docs/source/_static/pso_result.gif" alt="PSO 结果" height="150">
  </picture>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <picture>
    <source type="image/avif" srcset="docs/source/_static/rvea_result.avif">
    <img src="docs/source/_static/rvea_result.gif" alt="RVEA 结果" height="150">
  </picture>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <picture>
    <source type="image/avif" srcset="docs/source/_static/halfcheetah_200.avif">
    <img src="docs/source/_static/halfcheetah_200.gif" alt="HalfCheetah 200" height="150">
  </picture>
</p>

<div align="center">
  <a href="https://arxiv.org/abs/2301.12457"><img src="https://img.shields.io/badge/arxiv-2212.05652-red" alt="arXiv"></a>
  <a href="https://evox.readthedocs.io/zh_CN/latest/index.html"><img src="https://img.shields.io/badge/readthedocs-docs-green?logo=readthedocs" alt="文档"></a>
  <a href="https://pypi.org/project/evox/"><img src="https://img.shields.io/pypi/v/evox?logo=python" alt="PyPI 版本"></a>
  <a href="https://pypi.org/project/evox/"><img src="https://img.shields.io/badge/python-3.10+-orange?logo=python" alt="Python 版本"></a>
  <a href="https://discord.gg/Vbtgcpy7G4"><img src="https://img.shields.io/badge/discord-evox-%235865f2?logo=discord" alt="Discord 服务器"></a>
  <a href="https://qm.qq.com/q/vTPvoMUGAw"><img src="https://img.shields.io/badge/QQ-297969717-%231db4f4?logo=tencentqq" alt="QQ 群"></a>
</div>

<p align="center">
  <a href="./README.md"><img src="https://img.shields.io/badge/English-f6f5f4" alt="English README"></a>
  <a href="./README_ZH.md"><img src="https://img.shields.io/badge/中文-f6f5f4" alt="中文 README"></a>
</p>

---

<h3 align="center"> 🌟分布式 GPU 加速的通用演化计算框架🌟 </h3>

---

## 🔥 新闻
- [2025-05-13] 发布 **EvoX 1.2.2** - 🚀 EvoX v1.2.2 发布，新增 Mujoco Playground 及官方教程！ [[更新内容](https://evox.group/index.php?m=home&c=View&a=index&aid=151)]
- [2025-02-03] 发布 **EvoRL**：基于 **JAX** 的 GPU 加速 **进化强化学习** 框架！[[论文](https://arxiv.org/abs/2501.15129)] [[代码](https://github.com/EMI-Group/evorl)]
- [2025-01-30] 发布 **EvoGP**：基于 **PyTorch** & **CUDA** 的 GPU 加速 **遗传编程** 框架！[[论文](http://arxiv.org/abs/2501.17168)] [[代码](https://github.com/EMI-Group/evogp)]
- [2025-01-14] 发布 **EvoX 1.0.0**，全面兼容 **PyTorch**，全面接入`torch.compile`！使用 **JAX 版本** 的用户可在 **v0.9.0 分支** 获取。

## 目录

1. [概述](#概述)
2. [主要特性](#主要特性)
3. [主要内容](#主要内容)
4. [安装指南](#安装指南)
5. [快速开始](#快速开始)
6. [相关项目](#相关项目)
7. [社区支持](#社区支持)

## 概述

EvoX 是一个分布式 GPU 加速的进化计算框架，兼容 **PyTorch**。提供易用的编程模型，包含 **50+ 进化算法 (EAs)** 和 **100+ 基准问题/环境**。详情请参阅我们的 [论文](https://arxiv.org/abs/2301.12457) 及 [文档](https://evox.readthedocs.io/zh/latest/)。

> [!NOTE]
> 需要使用 **JAX 版本** 的用户请安装 **0.9.0**或更早版本。


## 主要特性

### 💻 高性能计算

#### 🚀 超高性能
- 支持在**CPU** 和 **GPU** 等异构硬件上加速运行，实现**100 倍以上加速**。
- 集成**分布式工作流**，可无缝扩展至多个节点或设备。

#### 🌐 一体化解决方案
- 内置**50+ 种算法**，全面支持**单目标和多目标优化**。
- 提供**分层架构**，适用于**元学习**、**超参数优化**和**神经进化**等复杂任务。

#### 🛠️ 易用设计
- **完全兼容 PyTorch** 及其生态系统，借助**定制化编程模型**简化算法开发。
- 具备**一键安装**功能，让 Windows 用户轻松上手。


### 📊 多功能基准测试

#### 📚 丰富的基准测试套件
- 提供**100+ 基准测试问题**，涵盖单目标优化、多目标优化及现实工程挑战。

#### 🎮 支持物理引擎
- 可无缝集成 **Brax** 等物理引擎，以及其他主流强化学习框架。

#### ⚙️ 可定制问题
- 提供**封装模块**，支持用户自定义问题，并可无缝集成到现实应用和数据集。


### 📈 灵活的可视化工具

#### 🔍 即用型工具
- 内置**多种可视化工具**，支持不同任务的进化过程分析。

#### 🛠️ 可定制模块
- 允许用户集成自定义**可视化代码**，提供灵活的展示方式。

#### 📂 实时数据流
- 采用定制的 **.exv 格式**，简化并加速**实时数据流处理**。

## 主要内容

<table border="1" cellspacing="0" cellpadding="8" style="border-collapse: collapse; width: 100%; text-align: left;">
  <thead>
    <tr>
      <th>类别</th>
      <th>子类别</th>
      <th>主要算法 / 基准问题</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3">单目标优化</td>
      <td><b>差分演化</b></td>
      <td>CoDE, JaDE, SaDE, SHADE, IMODE, ...</td>
    </tr>
    <tr>
      <td><b>演化策略</b></td>
      <td>CMA-ES, PGPE, OpenES, CR-FM-NES, xNES, ...</td>
    </tr>
    <tr>
      <td><b>粒子群优化</b></td>
      <td>FIPS, CSO, CPSO, CLPSO, SL-PSO, ...</td>
    </tr>
    <tr>
      <td rowspan="3">多目标优化</td>
      <td><b>基于支配关系</b></td>
      <td>NSGA-II, NSGA-III, SPEA2, BiGE, KnEA, ...</td>
    </tr>
    <tr>
      <td><b>基于分解策略</b></td>
      <td>MOEA/D, RVEA, t-DEA, MOEAD-M2M, EAG-MOEAD, ...</td>
    </tr>
    <tr>
      <td><b>基于指标</b></td>
      <td>IBEA, HypE, SRA, MaOEA-IGD, AR-MOEA, ...</td>
    </tr>
    <tr>
      <td rowspan="2">基准问题 / 评测环境</td>
      <td><b>数值优化问题</b></td>
      <td>DTLZ, LSMOP, MaF, ZDT, CEC'22, ...</td>
    </tr>
    <tr>
      <td><b>神经演化 / 强化学习</b></td>
      <td>Brax, TorchVision 数据集, ...</td>
    </tr>
  </tbody>
</table>

如需完整的算法列表及详细描述，请访问 [算法 API](https://evox.readthedocs.io/en/latest/apidocs/evox/evox.algorithms.html)，基准问题及环境请参考 [问题 API](https://evox.readthedocs.io/en/latest/apidocs/evox/evox.problems.html)。

## 安装指南

使用 `pip` 轻松安装包含基础功能的 `evox`：

```bash
pip install "evox[default]"
```

从源代码安装最新版本以进行测试或开发：

```bash
git clone https://github.com/EMI-Group/evox.git
cd evox
pip install -e .
```

> [!TIP]
> Windows 用户可使用 [win-install.bat](https://evox.readthedocs.io/en/latest/_downloads/796714545d73f0b52e921d885369323d/win-install.bat) 脚本安装。

## 快速开始

以下是一些示例，帮助你快速上手 EvoX：

### 单目标优化

使用 PSO 算法求解 Ackley 问题：

```python
import torch
from evox.algorithms import PSO
from evox.problems.numerical import Ackley
from evox.workflows import StdWorkflow, EvalMonitor

# torch.set_default_device("cuda") # 如果您希望默认使用 GPU，请取消注释此行

algorithm = PSO(pop_size=100, lb=-32 * torch.ones(10), ub=32 * torch.ones(10))
problem = Ackley()
monitor = EvalMonitor()
workflow = StdWorkflow(algorithm, problem, monitor)
workflow.init_step()
for i in range(100):
    workflow.step()

monitor.plot() # 或者在终端模式下调用 monitor.plot().show()
```

<details>
  <summary>样例输出</summary>

  <picture>
    <source type="image/avif" srcset="docs/source/_static/1-single-objective-output.avif">
    <img src="docs/source/_static/1-single-objective-output.png">
  </picture>

</details>

### 多目标优化

使用 RVEA 算法求解 DTLZ2 问题：

```python
import torch
from evox.algorithms import RVEA
from evox.metrics import igd
from evox.problems.numerical import DTLZ2
from evox.workflows import StdWorkflow, EvalMonitor

# torch.set_default_device("cuda") # 如果您希望默认使用 GPU，请取消注释此行

prob = DTLZ2(m=2)
pf = prob.pf()
algo = RVEA(
    pop_size=100,
    n_objs=2,
    lb=-torch.zeros(12),
    ub=torch.ones(12)
)
monitor = EvalMonitor()
workflow = StdWorkflow(algo, prob, monitor)
workflow.init_step()
for i in range(100):
    workflow.step()

monitor.plot() # 或者在终端模式下调用 monitor.plot().show()
```

<details>
  <summary>样例输出</summary>

  <picture>
    <source type="image/avif" srcset="docs/source/_static/2-multi-objective-output.avif">
    <img src="docs/source/_static/2-multi-objective-output.png">
  </picture>

</details>

### 神经演化

进化一个简单的 MLP 模型，求解 HalfCheetah 环境：

```python
import torch
import torch.nn as nn
from evox.algorithms import PSO
from evox.problems.neuroevolution.brax import BraxProblem
from evox.utils import ParamsAndVector
from evox.workflows import EvalMonitor, StdWorkflow

# torch.set_default_device("cuda") # 如果您希望默认使用 GPU，请取消注释此行

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 观察空间维度为 17，动作空间维度为 6。
        self.features = nn.Sequential(nn.Linear(17, 8), nn.Tanh(), nn.Linear(8, 6))

    def forward(self, x):
        return torch.tanh(self.features(x))

# 初始化 MLP 模型
model = SimpleMLP()
adapter = ParamsAndVector(dummy_model=model)
# 设置种群大小
POP_SIZE = 1024
# 获取 PSO 算法的边界
model_params = dict(model.named_parameters())
pop_center = adapter.to_vector(model_params)
lb = torch.full_like(pop_center, -5)
ub = torch.full_like(pop_center, 5)
# 初始化 PSO 算法，你也可以使用其他算法
algorithm = PSO(pop_size=POP_SIZE, lb=lb, ub=ub)
# 初始化 Brax 问题
problem = BraxProblem(
    policy=model,
    env_name="halfcheetah",
    max_episode_length=1000,
    num_episodes=3,
    pop_size=POP_SIZE,
)
# 设置监视器，可记录最佳 3 个适应度值
monitor = EvalMonitor(topk=3)
# 初始化工作流
workflow = StdWorkflow(
    algorithm=algorithm,
    problem=problem,
    monitor=monitor,
    opt_direction="max",
    solution_transform=adapter,
)
workflow.init_step()
for i in range(50):
    workflow.step()

monitor.plot() # 或者在终端模式下调用 monitor.plot().show()
```

<details>
  <summary>样例输出</summary>

  <picture>
    <source type="image/avif" srcset="docs/source/_static/3-neuroevolution-output.avif">
    <img src="docs/source/_static/3-neuroevolution-output.gif">
  </picture>

</details>

> [!NOTE]
> 如需完整指南，请访问我们的[文档](https://evox.readthedocs.io/zh/latest/)，其中包含详细的安装步骤、教程、实践示例以及完整的API说明。

## 相关项目

- **EvoRL**: 基于 GPU 加速的进化强化学习框架。查看详情：点击这里。
- **EvoGP**: 基于 GPU 加速的遗传编程框架。查看详情：点击这里。
- **TensorNEAT**: 用于 GPU 加速的张量化 NEAT（NeuroEvolution of Augmenting Topologies）框架。查看详情：点击这里。
- **TensorRVEA**: 用于 GPU 加速的张量化参考向量引导进化算法（RVEA）框架。查看详情：点击这里。
- **TensorACO**: 用于 GPU 加速的张量化蚁群优化算法（ACO）框架。查看详情：点击这里。
- **EvoXBench**: 一个用于解决各种优化问题（如神经架构搜索 NAS）的真实世界基准测试平台。该平台无需 GPU/PyTorch/TensorFlow 运行，并支持多种编程环境。查看详情：点击这里。

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
