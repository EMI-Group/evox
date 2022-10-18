from evoxlib import algorithms, problems, pipelines
import jax
import jax.numpy as jnp


# create a pipeline
pipeline = pipelines.StdPipeline(
    algorithms.PSO(
        lb=jnp.full(shape=(2,), fill_value=-32),
        ub=jnp.full(shape=(2,), fill_value=32),
        pop_size=100,
    ),
    problems.classic.Ackley(),
)

# init the pipeline
key = jax.random.PRNGKey(42)
state = pipeline.init(key)

# run the pipeline for 100 steps
for i in range(100):
    state = pipeline.step(state)
