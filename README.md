<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/source/_static/evox_logo_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/source/_static/evox_logo_light.png">
    <img alt="EvoX Logo" height="128" width="500px" src="docs/source/_static/evox_logo_light.png">
  </picture>
</h1>

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

<p align="center">
  <b>🌟Distributed GPU-accelerated Framework for Scalable Evolutionary Computation🌟</b>
</p>

---

##
Building upon [JAX](https://github.com/google/jax) and [Ray](https://github.com/ray-project/ray), EvoX offers a comprehensive suite of **50+ Evolutionary Algorithms (EAs)** and a wide range of **100+ Benchmark Problems**, all benefiting from distributed GPU-acceleration. It facilitates efficient exploration of complex optimization landscapes, effective tackling of black-box optimization challenges, and deep dives into neuroevolution with [Brax](https://github.com/google/brax). With a foundation in functional programming and hierarchical state management, EvoX offers a user-friendly and modular experience. For more details, please refer to our [Paper](https://arxiv.org/abs/2301.12457) and [Documentation](https://evox.readthedocs.io/).

---

## Key Features

- 🚀 **Fast Performance**:
  - Experience **GPU-Accelerated** optimization, achieving speeds 100x faster than traditional methods.
  - Leverage the power of **Distributed Workflows** for even more rapid optimization.

- 🌐 **Versatile Optimization Suite**:
  - Cater to all your needs with both **Single-objective** and **Multi-objective** optimization capabilities.
  - Dive into a comprehensive library of **Benchmark Problems**, ensuring robust testing and evaluation.
  - Explore the frontier of AI with extensive tools for **Neuroevolution** tasks.

- 🛠️ **Designed for Simplicity**:
  - Embrace the elegance of **Functional Programming**, simplifying complex algorithmic compositions.
  - Benefit from **Hierarchical State Management**, ensuring modular and clean programming.
  - Jumpstart your journey with our [Detailed Tutorial](https://evox.readthedocs.io/en/latest/guide/basics/index.html).

---

## Comprehensive Evolutionary Algorithms

### Single-Objective Optimization

| Category                    | Algorithm Names                             |
| --------------------------- | ------------------------------------------ |
| Differential Evolution      | CoDE, JaDE, SaDE, SHADE, IMODE, ...        |
| Evolution Strategies        | CMA-ES, PGPE, OpenES, CR-FM-NES, xNES, ... |
| Particle Swarm Optimization | FIPS, CSO, CPSO, CLPSO, SL-PSO, ...        |

### Multi-Objective Optimization

| Category           | Algorithm Names                                 |
| ------------------ | ---------------------------------------------- |
| Dominance-based    | NSGA-II, NSGA-III, SPEA2, BiGE, KnEA, ...      |
| Decomposition-based| MOEA/D, RVEA, t-DEA, MOEAD-M2M, EAG-MOEAD, ... |
| Indicator-based    | IBEA, HypE, SRA, MaOEA-IGD, AR-MOEA, ...       |

For a comprehensive list and further details of all algorithms, please check the [API Documentation](https://evox.readthedocs.io/en/latest/api/algorithms/index.html).

## Diverse Benchmark Problems

| Category      | Problem Names                           |
| ------------- | --------------------------------------- |
| Numerical     | DTLZ, LSMOP, MaF, ZDT, CEC'22,  ...    |
| Neuroevolution| Brax, Gym, TorchVision Dataset, ...    |

For a comprehensive list and further details of all benchmark problems, please check the [API Documentation](https://evox.readthedocs.io/en/latest/api/problems/index.html).


## Setting Up EvoX

Install `evox` effortlessly via `pip`:
```bash
pip install evox
```

**Note**: To install EvoX with JAX and hardware acceleration capabilities, please refer to our comprehensive [Installation Guide](https://evox.readthedocs.io/en/latest/guide/install.html).


## Quick Start

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

## Use-cases and Applications

Try out ready-to-play examples in your browser with Colab:

| Example | Link |
| ------- | ---- |
| Basic Usage | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EMI-Group/evox/blob/main/docs/source/guide/basics/1-start.ipynb) |
| Numerical Optimization | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EMI-Group/evox/blob/main/docs/source/example/pso_ackley.ipynb) |
| Neuroevolution with Gym | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EMI-Group/evox/blob/main/docs/source/example/gym_classic_control.ipynb) |
| Neuroevolution with Brax | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EMI-Group/evox/blob/main/docs/source/guide/basics/2-problems.ipynb) |
| Custom Algorithm/Problem | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EMI-Group/evox/blob/main/docs/source/example/custom_algorithm_and_problem.ipynb) |

For more use-cases and applications, pleae check out [Example Directory](https://evox.readthedocs.io/en/latest/example/index.html).

## Community & Support

- Engage in discussions and share your experiences on [GitHub Discussion Board](https://github.com/EMI-Group/evox/discussions).
- Join our QQ group (ID: 297969717).
- Help with the translation of the documentation on [Weblate](https://hosted.weblate.org/projects/evox/evox/).

![Translation status](https://hosted.weblate.org/widget/evox/evox/multi-auto.svg)

## Sister Projects

- EvoXBench: A benchmark platform for Neural Architecutre Search (NAS) without the requirement of GPUs/PyTorch/Tensorflow, supporting various programming languages such as Java, Matlab, Python, ect. Check out [here](https://github.com/EMI-Group/evoxbench).

## Citing EvoX

If you use EvoX in your research and want to cite it in your work, please use:
```
@article{evox,
  title = {{EvoX}: {A} {Distributed} {GPU}-accelerated {Framework} for {Scalable} {Evolutionary} {Computation}},
  author = {Huang, Beichen and Cheng, Ran and Li, Zhuozhao and Jin, Yaochu and Tan, Kay Chen},
  journal = {arXiv preprint arXiv:2301.12457},
  eprint = {2301.12457},
  year = {2023}
}
```



