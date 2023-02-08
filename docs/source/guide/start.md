# Getting Started

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