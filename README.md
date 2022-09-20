# Exmaple

```python
from evoxlib import algorithms, problems, pipelines
import jax
import jax.numpy as jnp


# create a pipeline
pipeline = pipelines.StdPipeline(
    algorithms.CSO(
        lb=jnp.full(shape=(2,), fill_value=-32),
        ub=jnp.full(shape=(2,), fill_value=32),
        pop_size=100,
    ),
    problems.classic.Ackley(),
    fitness_monitor=True,
    population_monitor=True
)

# init the pipeline
key = jax.random.PRNGKey(42)
state = pipeline.init(key)

# run the pipeline for 100 steps
for i in range(100):
    state = pipeline.step(state)

pipeline.fitness_monitor.show()
pipeline.population_monitor.show()

```