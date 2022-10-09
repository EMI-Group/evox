import evoxlib as exl
from evoxlib import pipelines, algorithms, problems
import jax
import jax.numpy as jnp
import pytest


def test_open_es():
    # create a pipeline
    pipeline = pipelines.StdPipeline(
        algorithm = algorithms.OpenES(
            init_params=jnp.full(shape=(10,), fill_value=1),
            pop_size=100,
            learning_rate=0.5,
            noise_std=0.5,
        ),
        problem=problems.classic.Ackley(),
        fitness_monitor=True
    )
    # init the pipeline
    key = jax.random.PRNGKey(42)
    state = pipeline.init(key)

    # run the pipeline for 100 steps
    for i in range(100):
        state = pipeline.step(state)

    # the result should be close to 0
    state, min_fitness = pipeline.get_min_fitness(state)
    assert min_fitness < 1e-4
