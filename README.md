<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/source/_static/evox_logo_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/source/_static/evox_logo_light.png">
    <img alt="EvoX Logo" height="128" width="500px" src="docs/source/_static/evox_logo_light.png">
  </picture>
</h1>

<p align="center">
  <img src="docs/source/_static/pso_result.gif" alt="PSO Result" height="150">  
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="docs/source/_static/rvea_result.gif" alt="RVEA Result" height="150">  
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="docs/source/_static/halfcheetah_200.gif" alt="HalfCheetah 200" height="150">
</p>


  [![arXiv](https://img.shields.io/badge/arxiv-2212.05652-red)](https://arxiv.org/abs/2301.12457)
  [![Documentation](https://img.shields.io/badge/readthedocs-docs-green?logo=readthedocs)](https://evox.readthedocs.io/)
  [![PyPI-Version](https://img.shields.io/pypi/v/evox?logo=python)](https://pypi.org/project/evox/)
  [![Python-Version](https://img.shields.io/badge/python-3.10+-orange?logo=python)](https://pypi.org/project/evox/)
  [![Discord Server](https://img.shields.io/badge/discord-evox-%235865f2?logo=discord)](https://discord.gg/Vbtgcpy7G4)
  [![QQ Group](https://img.shields.io/badge/QQ-297969717-%231db4f4?logo=tencentqq)](https://qm.qq.com/q/vTPvoMUGAw)
  <!--[![GitHub User's Stars](https://img.shields.io/github/stars/EMI-Group%2Fevox)](https://github.com/EMI-Group/evox)-->
    <!--[![PyPI-Downloads](https://img.shields.io/pypi/dm/evox?color=orange&logo=python)](https://pypi.org/project/evox/)-->

---

<h3 align="center"> 🌟Distributed GPU-accelerated Framework for Scalable Evolutionary Computation🌟 </h3>
<h3 align="center"> Fully Compatible with PyTorch / JAX</h3>


## 🔥 News

- [2025-01-13] Released **EvoX 1.0.0** 🎉, now fully compatible with **PyTorch** ! Users of the previous **JAX-based version** can access it on the **v0.9.0 branch**.

## Table of Contents

1. [Overview](#Overview)
2. [Key Features](#key-features)
3. [Main Contents](#main-contents)
4. [Quick Installation](#quick-installation)
5. [Sister Projects](#sister-projects)
6. [Community & Support](#community--support)
7. [Citing EvoX](#citing-evox)
8. [Star History](#star-history)

## Overview

EvoX is a distributed GPU-accelerated evolutionary computation framework compatible with **PyTorch***.  With a user-friendly programming model, it offers a comprehensive suite of **50+ Evolutionary Algorithms (EAs)** and a wide range of **100+ Benchmark Problems/Environments**. For more details, please refer to our [Paper](https://arxiv.org/abs/2301.12457) and [Documentation](https://evox.readthedocs.io/en/latest/) / [文档](https://evox.readthedocs.io/zh/latest/).
*Users of the previous **JAX-based version** can access it on the **v0.9.0 branch**.


## Key Features

- 🚀 **High-Performance**:
  - Harness the power of GPU-accelerated optimization to achieve **over 100x speedups** compared to traditional methods.
  - Scale your computations seamlessly with **distributed workflows** across multiple GPUs.
- 🌐 **Comprehensive and Versatile**:
  - Fully supports **Single-objective** and **Multi-objective** optimization needs.
  - Integrates advanced modules including **Neuroevolution**, **Reinforcement Learning (RL)**, and **Hyperparameter Optimization (HPO)**.
- 🛠️ **User-Friendly and Accessible**:
  - Designed for simplicity, with full **PyTorch compatibility** and an intuitive programming model that enhances developer productivity.
  - Effortless setup, including a **one-click installation script** for Windows users.


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

For a comprehensive list and detailed descriptions of all algorithms, please check the [Algorithms API](https://evox.readthedocs.io/en/latest/api/algorithms/index.html), and for benchmark problems/environments, refer to the [Problems API](https://evox.readthedocs.io/en/latest/api/problems/index.html).


## Quick Installation

Install `evox` effortlessly via `pip`:

```bash
pip install evox
```

**Note**: Windows users can use the [`win-install.bat`](https://github.com/EMI-Group/evox/blob/evoxtorch-main/win-install.bat) script for installation.

## Sister Projects

- **TensorNEAT**: Tensorized NeuroEvolution of Augmenting Topologies (NEAT) for GPU Acceleration. Check out [here](https://github.com/EMI-Group/tensorneat).
- **TensorRVEA**: Tensorized Reference Vector Guided Evolutionary Algorithm (RVEA) for GPU Acceleration. Check out [here](https://github.com/EMI-Group/tensorrvea).
- **TensorACO**: Tensorized Ant Colony Optimization (ACO) for GPU Acceleration. Check out [here](https://github.com/EMI-Group/tensoraco).
- **EvoXBench**: A real-world benchmark platform for solving various optimization problems, such as Neural Architecture Search (NAS). It operates without the need for GPUs/PyTorch/TensorFlow and supports multiple programming environments. Check out [here](https://github.com/EMI-Group/evoxbench).  

Stay tuned—more exciting developments are on the way!  ✨

## Community & Support

- Join discussions on the [GitHub Discussion Board](https://github.com/EMI-Group/evox/discussions).  
- Connect via [Discord](https://discord.gg/Vbtgcpy7G4) or QQ group (ID: 297969717).  
- Help translate EvoX docs on [Weblate](https://hosted.weblate.org/projects/evox/evox/).  
  We currently support translations in two languages, [English](https://evox.readthedocs.io/en/latest/) / [中文](https://evox.readthedocs.io/zh/latest/).
- Official Website: https://evox.group/


## Citing EvoX

If EvoX contributes to your research, please cite it:

```
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
