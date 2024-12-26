<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/source/_static/evox_logo_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/source/_static/evox_logo_light.png">
    <img alt="EvoX Logo" height="128" width="500px" src="docs/source/_static/evox_logo_light.png">
  </picture>
</h1>


  [![arXiv](https://img.shields.io/badge/arxiv-2212.05652-red)](https://arxiv.org/abs/2301.12457)
  [![Documentation](https://img.shields.io/badge/readthedocs-docs-green?logo=readthedocs)](https://evox.readthedocs.io/)
  [![PyPI-Version](https://img.shields.io/pypi/v/evox?logo=python)](https://pypi.org/project/evox/)
  [![Python-Version](https://img.shields.io/badge/python-3.9+-orange?logo=python)](https://pypi.org/project/evox/)
  [![Discord Server](https://img.shields.io/badge/discord-evox-%235865f2?logo=discord)](https://discord.gg/Vbtgcpy7G4)
  [![QQ Group](https://img.shields.io/badge/QQ-297969717-%231db4f4?logo=tencentqq)](https://qm.qq.com/q/vTPvoMUGAw)
  [![GitHub User's Stars](https://img.shields.io/github/stars/EMI-Group%2Fevox)](https://github.com/EMI-Group/evox)
    <!--[![PyPI-Downloads](https://img.shields.io/pypi/dm/evox?color=orange&logo=python)](https://pypi.org/project/evox/)-->

---

<h3 align="center">
  🌟Distributed GPU-accelerated Framework for Scalable Evolutionary Computation🌟
</h3>

---

Building upon [JAX](https://github.com/google/jax) and [Ray](https://github.com/ray-project/ray), EvoX offers a comprehensive suite of **50+ Evolutionary Algorithms (EAs)** and a wide range of **100+ Benchmark Problems/Environments**, all benefiting from distributed GPU-acceleration. It facilitates efficient exploration of complex optimization landscapes, effective tackling of black-box optimization challenges, and deep dives into neuroevolution with [Brax](https://github.com/google/brax). With a foundation in functional programming and hierarchical state management, EvoX offers a user-friendly and modular experience. For more details, please refer to our [Paper](https://arxiv.org/abs/2301.12457) and [Documentation](https://evox.readthedocs.io/en/latest/) / [文档](https://evox.readthedocs.io/zh/latest/).

## Key Features

- 🚀 **Fast Performance**:
  - Experience **GPU-Accelerated** optimization, achieving speeds over 100x faster than traditional methods.
  - Leverage the power of **Distributed Workflows** for even more rapid optimization.

- 🌐 **Versatile Optimization Suite**:
  - Cater to all your needs with both **Single-objective** and **Multi-objective** optimization capabilities.
  - Dive into a comprehensive library of **Benchmark Problems/Environments**, ensuring robust testing and evaluation.
  - Explore the frontier of AI with extensive tools for **Neuroevolution/RL** tasks.

- 🛠️ **Designed for Simplicity**:
  - Embrace the elegance of **Functional Programming**, simplifying complex algorithmic compositions.
  - Benefit from **Hierarchical State Management**, ensuring modular and clean programming.

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

For a comprehensive list and further details of all algorithms, please check the [API Documentation](https://evox.readthedocs.io/en/latest/api/algorithms/index.html).

### Benchmark Problems/Environments

| Category       | Problems/Environments               |
| -------------- | ----------------------------------- |
| Numerical      | DTLZ, LSMOP, MaF, ZDT, CEC'22,  ... |
| Neuroevolution/RL | Brax, Gym, TorchVision Dataset, ... |

For a comprehensive list and further details of all benchmark problems/environments, please check the [API Documentation](https://evox.readthedocs.io/en/latest/api/problems/index.html).


## Setting Up EvoX


## Prerequisites

- **Python**: Version 3.12 (or higher)
- **CUDA**: Version 12.1 (or higher)
- **PyTorch**: Version 2.5.0 (or higher recommended)

Install `evox` effortlessly via `pip`:
```bash
pip install evox
```

**Note**: To setup EvoX with **GPU acceleration** capabilities, you will need to setup **JAX** first. For detials, please refer to our comprehensive [Installation Guide](https://evox.readthedocs.io/en/latest/guide/install/index.html). Additionally, you can watch our **instructional videos**:

🎥 [EvoX Installation Guide (Linux)](https://youtu.be/fa2s1Jl-Fy0)

🎥 [EvoX Installation Guide (Windows)](https://youtu.be/7f8Uz1rqvn8)

🎥 [EvoX 安装指南 (Linux)](https://www.bilibili.com/video/BV1Zt421c7GN)

🎥 [EvoX 安装指南 (Windows)](https://www.bilibili.com/video/BV1Bb421h7bG)



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

| Example                  | Link                                                                                                                                                                                                   |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Basic Usage              | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EMI-Group/evox/blob/main/docs/source/guide/basics/1-start.ipynb)                 |
| Numerical Optimization   | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EMI-Group/evox/blob/main/docs/source/example/pso_ackley.ipynb)                   |
| Neuroevolution with Gym  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EMI-Group/evox/blob/main/docs/source/example/gym_classic_control.ipynb)          |
| Neuroevolution with Brax | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EMI-Group/evox/blob/main/docs/source/guide/basics/2-problems.ipynb)              |
| Custom Algorithm/Problem | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EMI-Group/evox/blob/main/docs/source/example/custom_algorithm_and_problem.ipynb) |

For more use-cases and applications, pleae check out [Example Directory](https://evox.readthedocs.io/en/latest/example/index.html).


## Unit Test Commands

```shell
python ./unit_test/algorithms/pso_variants/test_clpso.py
python ./unit_test/algorithms/pso_variants/test_cso.py
python ./unit_test/algorithms/pso_variants/test_dms_pso_el.py
python ./unit_test/algorithms/pso_variants/test_fs_pso.py
python ./unit_test/algorithms/pso_variants/test_pso.py
python ./unit_test/algorithms/pso_variants/test_sl_pso_gs.py
python ./unit_test/algorithms/pso_variants/test_sl_pso_us.py

python ./unit_test/core/test_jit_util.py 
python ./unit_test/core/test_module.py 

python ./unit_test/problems/test_hpo_wrapper.py 

python ./unit_test/utils/test_jit_fix.py 
python ./unit_test/utils/test_parameters_and_vector.py
python ./unit_test/utils/test_while.py 

python ./unit_test/workflows/test_std_workflow.py
```

## Community & Support

- Engage in discussions and share your experiences on [GitHub Discussion Board](https://github.com/EMI-Group/evox/discussions).
- Join our QQ group (ID: 297969717).
- Help with the translation of the documentation on [Weblate](https://hosted.weblate.org/projects/evox/evox/).
We currently support translations in two languages, [English](https://evox.readthedocs.io/en/latest/) / [中文](https://evox.readthedocs.io/zh/latest/).
- Official Website: https://evox.group/

## Sister Projects
- TensorNEAT: Tensorized NeuroEvolution of Augmenting Topologies (NEAT) for GPU Acceleration. Check out [here](https://github.com/EMI-Group/tensorneat).
- TensorRVEA: Tensorized Reference Vector Guided Evolutionary Algorithm (RVEA) for GPU Acceleration. Check out [here](https://github.com/EMI-Group/tensorrvea).
- TensorACO: Tensorized Ant Colony Optimization (ACO) for GPU Acceleration. Check out [here](https://github.com/EMI-Group/tensoraco).
- EvoXBench: A benchmark platform for Neural Architecutre Search (NAS) without the requirement of GPUs/PyTorch/Tensorflow, supporting various programming languages such as Java, Matlab, Python, ect. Check out [here](https://github.com/EMI-Group/evoxbench).

## Citing EvoX

If you use EvoX in your research and want to cite it in your work, please use:
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