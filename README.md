# EvoX: A JAX-based EC Library

### Brief Intro
During the past decades, evolutionary computation (EC) has demonstrated promising potential in
solving various complex optimization problems of relatively small scales. However, as modern 
EC research is facing problems of a much larger scale, computing power begins to struggle. To
meet this grave challenge, to present `EvoX`, a distributed GPU-accelerated EC library.
In this report, we will show how `EvoX` is suitable for all EC algorithms while extending
to many real-life benchmark problems. We will also show that `EvoX` can also greatly 
accelerate EC workflow.

### Simple Example

```python
from evox import algorithms, problems, pipelines
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