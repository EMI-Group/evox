<h1 align="center">
  <img src=./docs/source/_static/evox_logo.png alt="Logo" height="24em"/>
  <strong>EvoX</strong>
  <br>
  A Distributed GPU-accelerated Framework for Scalable Evolutionary Computation
</h1>

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


## Features

- Single-objective and multi-objective algorithms.
- GPU computing.
- Easy to use distributed pipeline.
- Support a wide range of problems.
- Hierarchical state managing.

### Index

- [Features](#features)
  - [Index](#index)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [More Tutorial](#more-tutorial)
- [Example](#example)
- [Support](#support)

## Installation

``
pip install evox
``

## Quick Start

To start with, import `evox`

```python
import evox
from evox import algorithms, problems, pipelines
```

Then, create an algorithm and a problem:

```python
pso = algorithms.PSO(
    lb=jnp.full(shape=(2,), fill_value=-32),
    ub=jnp.full(shape=(2,), fill_value=32),
    pop_size=100,
)
ackley = problems.numerical.Ackley()
```

The algorithm and the problem are composed together using `pipeline`:

```python
pipeline = pipelines.StdPipeline(pso, ackley)
```

To initialize the whole pipeline, call `init` on the pipeline object with a PRNGKey. Calling `init` will recursively initialize a tree of objects, meaning the algorithm pso and problem ackley are automatically initialize as well.

```python
key = jax.random.PRNGKey(42)
state = pipeline.init(key)
```

To run the pipeline, call `step` on the pipeline.

```python
# run the pipeline for 100 steps
for i in range(100):
    state = pipeline.step(state)
```

## More Tutorial

Head to our [tutorial page](https://evox.readthedocs.io/en/latest/guide/index.html).

## Example

The [example](https://github.com/EMI-Group/evox/tree/main/examples) folder has many examples on how to use EvoX.

## Support

- For general discussion, please head to Github's [discussion](https://github.com/EMI-Group/evox/discussions)
- For Chinese speakers, please consider to join the QQ group to discuss. (Group number: 297969717).
<img src="./docs/source/_static/qq_group_number.jpg" width="15%">

## Citation

```
@article{evox,
  title = {{EvoX}: {A} {Distributed} {GPU}-accelerated {Framework} for {Scalable} {Evolutionary} {Computation}},
  author = {Huang, Beichen and Cheng, Ran and Li, Zhuozhao and Jin, Yaochu and Tan, Kay Chen},
  journal = {arXiv preprint arXiv:2301.12457},
  eprint = {2301.12457},
  year = {2023}
}
```
