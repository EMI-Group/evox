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
    <img src="docs/source/_static/pso_result.gif" alt="PSO Result" height="150">
  </picture>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <picture>
    <source type="image/avif" srcset="docs/source/_static/rvea_result.avif">
    <img src="docs/source/_static/rvea_result.gif" alt="RVEA Result" height="150">
  </picture>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <picture>
    <source type="image/avif" srcset="docs/source/_static/halfcheetah_200.avif">
    <img src="docs/source/_static/halfcheetah_200.gif" alt="HalfCheetah 200" height="150">
  </picture>
</p>


<div align="center">
  <a href="https://arxiv.org/abs/2301.12457"><img src="https://img.shields.io/badge/arxiv-2212.05652-red" alt="arXiv"></a>
  <a href="https://evox.readthedocs.io/"><img src="https://img.shields.io/badge/readthedocs-docs-green?logo=readthedocs" alt="Documentation"></a>
  <a href="https://pypi.org/project/evox/"><img src="https://img.shields.io/pypi/v/evox?logo=python" alt="PyPI Version"></a>
  <a href="https://pypi.org/project/evox/"><img src="https://img.shields.io/badge/python-3.10+-orange?logo=python" alt="Python Version"></a>
  <a href="https://discord.gg/Vbtgcpy7G4"><img src="https://img.shields.io/badge/discord-evox-%235865f2?logo=discord" alt="Discord Server"></a>
  <a href="https://qm.qq.com/q/vTPvoMUGAw"><img src="https://img.shields.io/badge/QQ-297969717-%231db4f4?logo=tencentqq" alt="QQ Group"></a>
</div>

<p align="center">
  <a href="./README.md"><img src="https://img.shields.io/badge/English-f6f5f4" alt="English README"></a>
  <a href="./README_ZH.md"><img src="https://img.shields.io/badge/中文-f6f5f4" alt="中文 README"></a>
</p>

---

<h3 align="center"> 🌟Distributed GPU-accelerated Framework for Scalable Evolutionary Computation🌟 </h3>

---


## 🔥 News
- [2025-02-03] Released **EvoRL**: A GPU-accelerated framework for **Evolutionary Reinforcement Learning**, powered by **JAX** ! [[Paper](https://arxiv.org/abs/2501.15129)] [[Code](https://github.com/EMI-Group/evorl)]
- [2025-01-30] Released **EvoGP**: A GPU-accelerated framework for **Genetic Programming**, powered by **PyTorch** & **CUDA**! [[Paper](http://arxiv.org/abs/2501.17168)] [[Code](https://github.com/EMI-Group/evogp)]
- [2025-01-14] Released **EvoX 1.0.0** - now fully compatible with **PyTorch**! Users of the previous **JAX-based version** can access it on the **v0.9.0 branch**.

## Table of Contents

1. [Overview](#Overview)
2. [Key Features](#key-features)
3. [Main Contents](#main-contents)
4. [Quick Installation](#quick-installation)
5. [Sister Projects](#sister-projects)
6. [Community & Support](#community--support)

## Overview

EvoX is a distributed GPU-accelerated evolutionary computation framework compatible with **PyTorch***.  With a user-friendly programming model, it offers a comprehensive suite of **50+ Evolutionary Algorithms (EAs)** and a wide range of **100+ Benchmark Problems/Environments**. For more details, please refer to our [Paper](https://arxiv.org/abs/2301.12457) and [Documentation](https://evox.readthedocs.io/en/latest/) / [文档](https://evox.readthedocs.io/zh/latest/).

*Users of the previous **JAX-based version** can access it on the **v0.9.0 branch**.


## Key Features

### 💻 High-Performance Computing

#### 🚀 Ultra Performance
- Supports acceleration on heterogeneous hardware, including both **CPUs** and **GPUs**, achieving over **100x speedups**.
- Integrates **distributed workflows** that scale seamlessly across multiple nodes or devices.

#### 🌐 All-in-One Solution
- Includes **50+ algorithms** for a wide range of use cases, fully supporting **single- and multi-objective optimization**.
- Provides a **hierarchical architecture** for complex tasks such as **meta learning**, **hyperparameter optimization**, and **neuroevolution**.

#### 🛠️ Easy-to-Use Design
- Fully compatible with **PyTorch** and its ecosystem, simplifying algorithmic development with a **tailored programming model**.
- Ensures effortless setup with **one-click installation** for Windows users.


### 📊 Versatile Benchmarking

#### 📚 Extensive Benchmark Suites
- Features **100+ benchmark problems** spanning single-objective optimization, multi-objective optimization, and real-world engineering challenges.

#### 🎮 Support for Physics Engines
- Integrates seamlessly with physics engines like **Brax** and other popular frameworks for reinforcement learning.

#### ⚙️ Customizable Problems
- Provides an **encapsulated module** for defining and evaluating custom problems tailored to user needs, with seamless integration into real-world applications and datasets.


### 📈 Flexible Visualization

#### 🔍 Ready-to-Use Tools
- Offers a comprehensive set of **visualization tools** for analyzing evolutionary processes across various tasks.

#### 🛠️ Customizable Modules
- Enables users to integrate their own **visualization code**, allowing for tailored and flexible visualizations.

#### 📂 Real-Time Data Streaming
- Leverages the tailored **.exv format** to simplify and accelerate real-time data streaming.



## Main Contents

### Evolutionary Algorithms for Single-objective Optimization

| Category                    | Algorithms                                 |
| --------------------------- | ------------------------------------------ |
| Differential Evolution      | CoDE, JaDE, SaDE, SHADE, IMODE, ...        |
| Evolution Strategy          | CMA-ES, PGPE, OpenES, CR-FM-NES, xNES, ... |
| Particle Swarm Optimization | FIPS, CSO, CPSO, CLPSO, SL-PSO, ...        |

### Evolutionary Algorithms for Multi-objective Optimization

| Category            | Algorithms                                     |
| ------------------- | ---------------------------------------------- |
| Dominance-based     | NSGA-II, NSGA-III, SPEA2, BiGE, KnEA, ...      |
| Decomposition-based | MOEA/D, RVEA, t-DEA, MOEAD-M2M, EAG-MOEAD, ... |
| Indicator-based     | IBEA, HypE, SRA, MaOEA-IGD, AR-MOEA, ...       |

### Benchmark Problems/Environments

| Category          | Problems/Environments               |
| ----------------- | ----------------------------------- |
| Numerical         | DTLZ, LSMOP, MaF, ZDT, CEC'22,  ... |
| Neuroevolution/RL | Brax, TorchVision Dataset, ...      |

For a comprehensive list and detailed descriptions of all algorithms, please check the [Algorithms API](https://evox.readthedocs.io/en/latest/apidocs/evox/evox.algorithms.html), and for benchmark problems/environments, refer to the [Problems API](https://evox.readthedocs.io/en/latest/apidocs/evox/evox.problems.html).


## Quick Installation

Install `evox` effortlessly via `pip`:

```bash
pip install evox
```

**Note**: Windows users can use the [win-install.bat](https://evox.readthedocs.io/en/latest/_downloads/796714545d73f0b52e921d885369323d/win-install.bat) script for installation.

## Sister Projects
- **EvoRL**:GPU-accelerated framework for Evolutionary Reinforcement Learning. Check out [here](https://github.com/EMI-Group/evorl).
- **EvoGP**:GPU-accelerated framework for Genetic Programming. Check out [here](https://github.com/EMI-Group/evogp).
- **TensorNEAT**: Tensorized NeuroEvolution of Augmenting Topologies (NEAT) for GPU Acceleration. Check out [here](https://github.com/EMI-Group/tensorneat).
- **TensorRVEA**: Tensorized Reference Vector Guided Evolutionary Algorithm (RVEA) for GPU Acceleration. Check out [here](https://github.com/EMI-Group/tensorrvea).
- **TensorACO**: Tensorized Ant Colony Optimization (ACO) for GPU Acceleration. Check out [here](https://github.com/EMI-Group/tensoraco).
- **EvoXBench**: A real-world benchmark platform for solving various optimization problems, such as Neural Architecture Search (NAS). It operates without the need for GPUs/PyTorch/TensorFlow and supports multiple programming environments. Check out [here](https://github.com/EMI-Group/evoxbench).

Stay tuned - more exciting developments are on the way!  ✨

## Community & Support

- Join discussions on the [GitHub Discussion Board](https://github.com/EMI-Group/evox/discussions).
- Connect via [Discord](https://discord.gg/Vbtgcpy7G4) or QQ group (ID: 297969717).
- Help translate EvoX docs on [Weblate](https://hosted.weblate.org/projects/evox/evox/).
  We currently support translations in two languages, [English](https://evox.readthedocs.io/en/latest/) / [中文](https://evox.readthedocs.io/zh/latest/).


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

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=EMI-Group/evox&type=Date)](https://star-history.com/#EMI-Group/evox&Date)
