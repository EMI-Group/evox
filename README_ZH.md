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
  <a href="https://evox.readthedocs.io/zh/latest/"><img src="https://img.shields.io/badge/readthedocs-docs-green?logo=readthedocs" alt="文档"></a>
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
- [2025-03-01] 发布 **EvoX 1.1.0** - 全面支持 `torch.compile` (TorchDynamo) [[Details](https://evox.group/index.php?m=home&c=View&a=index&aid=147)]
- [2025-02-03] 发布 **EvoRL**：基于 **JAX** 的 GPU 加速 **进化强化学习** 框架！[[论文](https://arxiv.org/abs/2501.15129)] [[代码](https://github.com/EMI-Group/evorl)]
- [2025-01-30] 发布 **EvoGP**：基于 **PyTorch** & **CUDA** 的 GPU 加速 **遗传编程** 框架！[[论文](http://arxiv.org/abs/2501.17168)] [[代码](https://github.com/EMI-Group/evogp)]
- [2025-01-14] 发布 **EvoX 1.0.0**，全面兼容 **PyTorch**，全面接入`torch.compile`！使用 **JAX 版本** 的用户可在 **v0.9.0 分支** 获取。

## 目录

1. [概述](#概述)
2. [主要特性](#主要特性)
3. [主要内容](#主要内容)
4. [快速安装](#快速安装)
5. [相关项目](#相关项目)
6. [社区 & 支持](#社区--支持)

## 概述

EvoX 是一个分布式 GPU 加速的进化计算框架，兼容 **PyTorch**。提供易用的编程模型，包含 **50+ 进化算法 (EAs)** 和 **100+ 基准问题/环境**。详情请参阅我们的 [论文](https://arxiv.org/abs/2301.12457) 及 [文档](https://evox.readthedocs.io/zh/latest/)。

*使用 **JAX 版本** 的用户可在 **v0.9.0 分支** 获取。*

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

### 用于单目标优化的进化算法

| 类别                      | 算法                                         |
| ------------------------- | -------------------------------------------- |
| 差分进化 (Differential Evolution) | CoDE, JaDE, SaDE, SHADE, IMODE, ...        |
| 进化策略 (Evolution Strategy)   | CMA-ES, PGPE, OpenES, CR-FM-NES, xNES, ... |
| 粒子群优化 (Particle Swarm Optimization) | FIPS, CSO, CPSO, CLPSO, SL-PSO, ...        |

### 用于多目标优化的进化算法

| 类别              | 算法                                           |
| ---------------- | ---------------------------------------------- |
| 基于支配关系 (Dominance-based)     | NSGA-II, NSGA-III, SPEA2, BiGE, KnEA, ...      |
| 基于分解策略 (Decomposition-based) | MOEA/D, RVEA, t-DEA, MOEAD-M2M, EAG-MOEAD, ... |
| 基于指标 (Indicator-based)     | IBEA, HypE, SRA, MaOEA-IGD, AR-MOEA, ...       |

### 基准测试问题/环境

| 类别              | 问题/环境                                   |
| ---------------- | ----------------------------------------- |
| 数值优化 (Numerical)         | DTLZ, LSMOP, MaF, ZDT, CEC'22, ... |
| 神经进化/强化学习 (Neuroevolution/RL) | Brax, TorchVision 数据集, ...      |

要查看所有算法的完整列表及详细描述，请访问 [算法 API](https://evox.readthedocs.io/en/latest/apidocs/evox/evox.algorithms.html)。
要查看基准测试问题/环境，请参考 [问题 API](https://evox.readthedocs.io/en/latest/apidocs/evox/evox.problems.html)。

## 快速安装

使用 `pip` 轻松安装 `evox`：

```bash
pip install evox
```

**注意**：Windows 用户可使用 [win-install.bat](https://evox.readthedocs.io/en/latest/_downloads/796714545d73f0b52e921d885369323d/win-install.bat) 脚本安装。

## 相关项目

- **EvoRL**: 基于 GPU 加速的进化强化学习框架。查看详情：点击这里。
- **EvoGP**: 基于 GPU 加速的遗传编程框架。查看详情：点击这里。
- **TensorNEAT**: 用于 GPU 加速的张量化 NEAT（NeuroEvolution of Augmenting Topologies）框架。查看详情：点击这里。
- **TensorRVEA**: 用于 GPU 加速的张量化参考向量引导进化算法（RVEA）框架。查看详情：点击这里。
- **TensorACO**: 用于 GPU 加速的张量化蚁群优化算法（ACO）框架。查看详情：点击这里。
- **EvoXBench**: 一个用于解决各种优化问题（如神经架构搜索 NAS）的真实世界基准测试平台。该平台无需 GPU/PyTorch/TensorFlow 运行，并支持多种编程环境。查看详情：点击这里。

敬请期待——更多精彩内容即将推出！✨

## 社区与支持

- 在 [GitHub 讨论区](https://github.com/EMI-Group/evox/discussions) 参与讨论。
- 通过 [Discord](https://discord.gg/Vbtgcpy7G4) 或 QQ 群（ID: 297969717）联系交流。
- 在 [Weblate](https://hosted.weblate.org/projects/evox/evox/) 帮助翻译 EvoX 文档。
  我们目前支持两种语言的翻译：[English](https://evox.readthedocs.io/en/latest/) / [中文](https://evox.readthedocs.io/zh/latest/)。
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

## Star 历史

[![Star 历史图表](https://api.star-history.com/svg?repos=EMI-Group/evox&type=Date)](https://star-history.com/#EMI-Group/evox&Date)
