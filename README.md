<p align="center">
  <img src="https://raw.githubusercontent.com/EMI-Group/evox/main/docs/source/_static/evox_logo_with_title.svg" width="500px" alt="EvoX Logo"/>
</p>

<h3 align="center">
  <a href="https://arxiv.org/abs/2301.12457">📄 Paper</a> |
  <a href="https://evox.readthedocs.io/">📚 Documentation</a> |
  <a href="https://github.com/EMI-Group/evox/actions/workflows/python-package.yml">🛠️ Build Status</a>
</h3>

<p align="center">
  <a href="https://arxiv.org/abs/2301.12457">
    <img src="https://img.shields.io/badge/paper-arxiv-red?style=for-the-badge" alt="EvoX Paper on arXiv">
  </a>

  <a href="https://evox.readthedocs.io/">
    <img src="https://img.shields.io/badge/docs-readthedocs-blue?style=for-the-badge" alt="EvoX Documentation">
  </a>

  <a href="https://github.com/EMI-Group/evox/actions/workflows/python-package.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/EMI-Group/evox/python-package.yml?style=for-the-badge" alt="EvoX Build Status">
  </a>
</p>

---

🌟 Experience the transformative power of **Distributed GPU-Acceleration** in **Evolutionary Computation (EC)**. EvoX isn't just another framework—it's a pioneering toolset crafted to **redefine EC's frontiers**. Dive deep into a vast collection of **Evolutionary Algorithms (EAs)** and engage with an expansive range of **Benchmark Problems**. Tackle everything from intricate tasks to computationally intensive challenges. With EvoX, achieve unmatched speed and adaptability, ensuring your optimization journey is swift and seamless. Embrace the future of EC with EvoX!

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

**Elevate Your Optimization Game with EvoX!**: Step into a meticulously crafted platform tailored for both researchers and enthusiasts. Effortlessly traverse the vast **optimization landscapes**, confront and conquer widely-acknowledged **black-box optimization challenges**, and venture into the intricate realms of **neuroevolution**. It's not merely about breadth—it's about velocity. Supercharge your projects with **GPU acceleration** and streamlined **distributed workflows**. Plus, with a foundation in **functional programming** and **hierarchical state management**, EvoX promises a seamless, modular user experience.

## 🧬 Comprehensive Evolutionary Algorithms

### 🎯 Single-Objective Optimization

| Category                    | Algorithm Names                             |
| --------------------------- | ------------------------------------------ |
| Differential Evolution      | CoDE, JaDE, SaDE, SHADE, IMODE, ...        |
| Evolution Strategies        | CMA-ES, PGPE, OpenES, CR-FM-NES, xNES, ... |
| Particle Swarm Optimization | FIPS, CSO, CPSO, CLPSO, SL-PSO, ...        |

### 🌐 Multi-Objective Optimization

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

Install `evox` effortlessly via `pip`:
```bash
pip install evox
```

**Note**: To install EvoX with JAX and hardware acceleration capabilities, please refer to our comprehensive [installation guide](https://evox.readthedocs.io/en/latest/guide/install.html).


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

Eager to delve deeper? The [example directory](https://evox.readthedocs.io/en/latest/example/index.html) is brimming with comprehensive use-cases and applications of EvoX.

## 🤝 Join the EvoX Community

- Engage in enlightening discussions and share your experiences on GitHub's [discussion board](https://github.com/EMI-Group/evox/discussions).
- Welcome to join our **QQ group** (ID: 297969717).

## Translation

We use weblate for translation, to help us translate the document, please visit [here](https://hosted.weblate.org/projects/evox/evox/).

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

<p align="center">
  ❤️ Found EvoX helpful? Please consider giving it a star to show your support! ⭐
</p>


