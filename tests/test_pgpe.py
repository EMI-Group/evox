import evox as ex
from evox import algorithms, problems, pipelines
from evox.monitors import FitnessMonitor
import jax
import jax.numpy as jnp
import pytest


def test_pgpe():
    monitor = FitnessMonitor()
    # create a pipeline
    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)
    center = jax.random.uniform(key1, shape=(2, ), minval=-5, maxval=5)
    pipeline = pipelines.StdPipeline(
        algorithm=ex.algorithms.PGPE(
            pop_size=100,
            center_init=center,
            optimizer='adam',
            center_learning_rate=0.3,
            stdev_init=1,
            stdev_learning_rate=0.1
        ),
        problem=ex.problems.classic.Rastrigin(),
        fitness_transform=monitor.update
    )
    # init the pipeline
    state = pipeline.init(key2)

    # run the pipeline for 100 steps
    for i in range(1000):
        state = pipeline.step(state)

    # the result should be close to 0
    min_fitness = monitor.get_min_fitness()
    assert min_fitness < 1e-1
