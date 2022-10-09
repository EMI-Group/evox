import evoxlib as exl
from evoxlib import algorithms, problems, pipelines
from evoxlib.monitors import FitnessMonitor
import jax
import jax.numpy as jnp
import pytest


def test_clustered_cso():
    # create a pipeline
    monitor = FitnessMonitor()
    pipeline = pipelines.StdPipeline(
        algorithms.ClusterdAlgorithm(
            base_algorithm=exl.algorithms.CSO(
                lb=jnp.full(shape=(10,), fill_value=-32),
                ub=jnp.full(shape=(10,), fill_value=32),
                pop_size=100,
            ),
            dim=100,
            num_cluster=10,
        ),
        problem=problems.classic.Ackley(),
        fitness_transform=monitor.update
    )
    # init the pipeline
    key = jax.random.PRNGKey(42)
    state = pipeline.init(key)

    # run the pipeline for 300 steps
    for i in range(300):
        state = pipeline.step(state)

    min_fitness = monitor.get_min_fitness()
    assert min_fitness < 1


def test_random_mask_cso():
    # create a pipeline
    monitor = FitnessMonitor()
    pipeline = pipelines.StdPipeline(
        algorithms.RandomMaskAlgorithm(
            base_algorithm=exl.algorithms.CSO(
                lb=jnp.full(shape=(10,), fill_value=-32),
                ub=jnp.full(shape=(10,), fill_value=32),
                pop_size=100,
            ),
            dim=100,
            num_cluster=10,
            num_mask=2,
            change_every=10,
        ),
        problem=problems.classic.Ackley(),
        fitness_transform=monitor.update
    )
    # init the pipeline
    key = jax.random.PRNGKey(42)
    state = pipeline.init(key)

    # run the pipeline for 300 steps
    for i in range(600):
        state = pipeline.step(state)

    min_fitness = monitor.get_min_fitness()
    assert min_fitness < 1
