<p align="center">
  <img src="https://raw.githubusercontent.com/EMI-Group/evox/main/docs/source/_static/evox_logo_with_title.svg" width="500px" alt="EvoX Logo"/>
</p>
<div align="center">
  <a href="https://evox.readthedocs.io/">
    <img src="https://img.shields.io/badge/docs-readthedocs-blue?style=for-the-badge" href="https://evox.readthedocs.io/">
  </a>
  <a href="https://arxiv.org/abs/2301.12457">
    <img src="https://img.shields.io/badge/paper-arxiv-red?style=for-the-badge">
  </a>
  <a href="https://github.com/EMI-Group/evox/actions/workflows/python-package.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/EMI-Group/evox/python-package.yml?style=for-the-badge">
  </a>
</div>

---

🌟 **Introducing EvoX**: Unleash the power of **Distributed GPU-Acceleration** in your Evolutionary Computation projects! With EvoX, you're not just getting a framework; you're accessing a state-of-the-art tool designed to redefine the limits of EC. Dive into a rich library packed with a myriad of Evolutionary Algorithms and an extensive range of benchmark problems. From intricate to computationally demanding tasks, EvoX is engineered to deliver unparalleled speed and adaptability, making your optimization journey swift and seamless. Experience the next-gen of EC with EvoX!

---

## ⭐️ Highlighted Features

- 🚀 **Blazing Fast Performance**:
  - Experience **GPU-Accelerated** optimization, achieving speeds 10x-100x faster than traditional methods.
  - Leverage the power of distributed workflows for even more rapid optimization.
  
- 🌐 **Versatile Optimization Suite**:
  - Cater to all your needs with both **Single-objective** and **Multi-objective** optimization capabilities.
  - Dive into a comprehensive library of benchmark problems, ensuring robust testing and evaluation.
  - Explore the frontier of AI with extensive tools for **neuroevolution** tasks.
  
- 🛠️ **Designed for Simplicity**:
  - Embrace the elegance of functional programming, simplifying complex algorithmic compositions.
  - Benefit from hierarchical state management, ensuring modular and clean programming.
  - Jumpstart your journey with our [detailed tutorial](https://evox.readthedocs.io/en/latest/guide/index.html).

**Elevate Your Optimization Game with EvoX!** Dive into an optimization powerhouse, meticulously crafted to empower both researchers and practitioners. With EvoX, navigating the vast terrains of optimization becomes not just feasible, but effortlessly intuitive. Whether you're dealing with widely recognized benchmark challenges or venturing into the intricate realms of neuroevolution, EvoX stands as your versatile experimentation platform. But it's not just about variety – it's about speed. Harness the might of GPU acceleration and distributed workflows to conquer even the most computationally intensive tasks. And, thanks to its foundation in functional programming and hierarchical state management, you're guaranteed a user experience that champions both efficiency and modularity.

### 📑 Table of Contents

- [🧬 Comprehensive Evolutionary Algorithms](#-comprehensive-evolutionary-algorithms)
    - [🎯 Single-Objective Algorithms](#-single-objective-algorithms)
    - [🌐 Multi-Objective Algorithms](#-multi-objective-algorithms)
- [📊 Diverse Benchmark Problems](#-diverse-benchmark-problems)
- [🔧 Installation](#-setting-up-evox)
- [🚀 Getting Started](#-dive-right-in-quick-start)
- [🔍 Examples](#-explore-more-with-examples)
- [🤝 Community & Support](#-join-the-evox-community)
- [📝 How to Cite EvoX](#-citing-evox)
- 
## 🧬 Comprehensive Evolutionary Algorithms

### 🎯 Single-Objective Algorithms

| Category                    | Algorithm Names                             |
| --------------------------- | ------------------------------------------ |
| Differential Evolution      | CoDE, JaDE, SaDE, SHADE, IMODE, ...        |
| Evolution Strategies        | CMA-ES, PGPE, OpenES, CR-FM-NES, xNES, ... |
| Particle Swarm Optimization | FIPS, CSO, CPSO, CLPSO, SL-PSO, ...        |

### 🌐 Multi-Objective Algorithms

| Category           | Algorithm Names                                 |
| ------------------ | ---------------------------------------------- |
| Dominance-based    | NSGA-II, NSGA-III, SPEA2, BiGE, KnEA, ...      |
| Decomposition-based| MOEA/D, RVEA, t-DEA, MOEAD-M2M, EAG-MOEAD, ... |
| Indicator-based    | IBEA, HypE, SRA, MaOEA-IGD, AR-MOEA, ...       |

## 📊 Diverse Benchmark Problems

| Category      | Problem Names                           |
| ------------- | --------------------------------------- |
| Numerical     | DTLZ, LSMOP, MaF, ZDT, CEC'22,  ...    |
| Neuroevolution| Brax, Gym, TorchVision Dataset, ...    |

Dive deeper! For a comprehensive list and further details, explore our [API Documentation](https://evox.readthedocs.io/en/latest/api/algorithms/index.html) for algorithms and [Benchmark Problems](https://evox.readthedocs.io/en/latest/api/problems/index.html).

## 🔧 Setting Up EvoX

**Step 1**: Install `evox` effortlessly via `pip`:
```bash
pip install evox
```
**Note**: EvoX thrives on the power of JAX. Ensure you have JAX set up by following the [official JAX installation guide](https://github.com/google/jax?tab=readme-ov-file#installation).

## 🚀 Dive Right In: Quick Start

Kickstart your journey with EvoX in just a few simple steps:
1. **Import necessary modules**:
```python
import evox
from evox import algorithms, problems, workflows
```
2. **Configure an algorithm and define a problem**:
```python
pso = algorithms.PSO(
    lb=jnp.full(shape=(2,), fill_value=-32),
    ub=jnp.full(shape=(2,), fill_value=32),
    pop_size=100,
)
ackley = problems.numerical.Ackley()
```
3. **Compose and initialize the workflow**:
```python
workflow = workflows.StdWorkflow(pso, ackley)
key = jax.random.PRNGKey(42)
state = workflow.init(key)
```
4. **Run the workflow**:
```python
# Execute the workflow for 100 iterations
for i in range(100):
    state = workflow.step(state)
```

## 🔍 Explore More with Examples

Eager to delve deeper? The [example directory](https://github.com/EMI-Group/evox/tree/main/examples) is brimming with comprehensive use-cases and applications of EvoX.

## 🤝 Join the EvoX Community

- Engage in enlightening discussions and share your experiences on GitHub's [discussion board](https://github.com/EMI-Group/evox/discussions).
- For our Chinese enthusiasts, we welcome you to our **QQ group** (ID: 297969717).

## 📝 Citing EvoX

If EvoX has propelled your research or projects, consider citing our work:
```
@article{evox,
  title = {{EvoX}: {A} {Distributed} {GPU}-accelerated {Framework} for {Scalable} {Evolutionary} {Computation}},
  author = {Huang, Beichen and Cheng, Ran and Li, Zhuozhao and Jin, Yaochu and Tan, Kay Chen},
  journal = {arXiv preprint arXiv:2301.12457},
  eprint = {2301.12457},
  year = {2023}
}
```
