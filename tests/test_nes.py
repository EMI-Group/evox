from evoxlib import pipelines, algorithms, problems
import jax
import jax.numpy as jnp
import pytest


def test_xnes():
    # create a pipeline
    pipeline = pipelines.StdPipeline(
        algorithm = algorithms.xNES(
            init_mean=jnp.full(shape=(100,), fill_value=1),
            init_covar=jnp.eye(100) * 2,
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

    
def test_snes():
    # create a pipeline
    pipeline = pipelines.StdPipeline(
        algorithm = algorithms.SeparableNES(
            init_mean=jnp.full(shape=(100,), fill_value=1),
            init_std=jnp.full(shape=(100,), fill_value=10),
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