# Getting Started

## Algorithm, Problem & Pipeline

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

## Monitor

Usually, we don't care about the final state of the pipeline, instead, we are more interested in things like the fitness or the final solution.

The `monitor` is the way to record these values in EvoX.

First, import monitos and create a monitor

```python
from evox.monitors import FitnessMonitor
monitor = FitnessMonitor()
```

Then set this monitor as the fitness transform for the pipeline

```python
pipeline = pipelines.StdPipeline(
    pso,
    ackley,
    fitness_transform=monitor.update,
)
```

Then continue to run the pipeline as ususal. now at each iteration, the pipeline will call `monitor.update` with the fitness at that iteration.

```python
# run the pipeline for 100 steps
for i in range(100):
    state = pipeline.step(state)
```

To get the minimum fitness of all time, call the `get_min_fitness` method on the monitor.

```python
# print the min fitness
print(monitor.get_min_fitness())
```