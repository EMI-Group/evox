# EvoX: A Distributed GPU-accelerated Library for Evolutionary Computation

<h4 align="left">
  [<a href="https://evox.readthedocs.io/">Docs</a>]
  [<a href="https://arxiv.org/abs/2301.12457">Paper</a>]
</h4>

## Features

- Single-objective and multi-objective algorithms.
- GPU computing.
- Easy to use distributed pipeline.
- Support a wide range of problems.

### Index

- [Getting started](#getting-started)
- [Example](#exmaple)

## Installation

``
pip install evox
``

## Getting started

To start with, import `evox`

```python
import evox
from evox import algorithm, problem, pipeline
```

Then, create an algorithm and a problem:

```python
pso = algorithms.PSO(
    lb=jnp.full(shape=(2,), fill_value=-32),
    ub=jnp.full(shape=(2,), fill_value=32),
    pop_size=100,
)
ackley = problems.classic.Ackley()
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

For more detailed usage, please refer to our [documentation](https://evox.readthedocs.io/).

## Exmaple

The [example](https://github.com/EMI-Group/evox/tree/main/examples) folder has many examples on how to use EvoX.