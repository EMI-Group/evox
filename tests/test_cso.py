import evoxlib as exl
from evoxlib import pipelines, algorithms, problems
from evoxlib.monitors import FitnessMonitor
import jax
import jax.numpy as jnp
import pytest


def test_cso():
    monitor = FitnessMonitor()
    # create a pipelines
    pipeline = pipelines.StdPipeline(
        algorithm=algorithms.CSO(
            lb=jnp.full(shape=(2,), fill_value=-32),
            ub=jnp.full(shape=(2,), fill_value=32),
            pop_size=100,
        ),
        problem=problems.classic.Ackley(),
        fitness_transform=monitor.update
    )
    # init the pipeline
    key = jax.random.PRNGKey(42)
    state = pipeline.init(key)

    # run the pipeline for 100 steps
    for i in range(100):
        state = pipeline.step(state)

    # the result should be close to 0
    min_fitness = monitor.get_min_fitness()
    assert min_fitness < 1e-4
