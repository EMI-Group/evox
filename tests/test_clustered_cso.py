import evox as ex
import jax
import jax.numpy as jnp
import pytest
from evox import algorithms, pipelines, problems
from evox.monitors import FitnessMonitor
import chex


@pytest.mark.parametrize("num_gpus", [None, 1])
def test_clustered_cso(num_gpus):
    # create a pipeline
    monitor = FitnessMonitor()
    pipeline = pipelines.StdPipeline(
        algorithms.ClusterdAlgorithm(
            base_algorithm=ex.algorithms.CSO(
                lb=jnp.full(shape=(10,), fill_value=-32),
                ub=jnp.full(shape=(10,), fill_value=32),
                pop_size=100,
            ),
            dim=40,
            num_cluster=4,
            num_gpus=num_gpus
        ),
        problem=problems.classic.Ackley(),
        fitness_transform=monitor.update
    )
    # init the pipeline
    key = jax.random.PRNGKey(42)
    state = pipeline.init(key)

    # run the pipeline for 50 steps
    for i in range(50):
        state = pipeline.step(state)

    min_fitness = monitor.get_min_fitness()
    assert abs(min_fitness - 10.785286903381348) < 1e-4


def test_random_mask_cso():
    # create a pipeline
    monitor = FitnessMonitor()
    pipeline = pipelines.StdPipeline(
        algorithms.RandomMaskAlgorithm(
            base_algorithm=ex.algorithms.CSO(
                lb=jnp.full(shape=(10,), fill_value=-32),
                ub=jnp.full(shape=(10,), fill_value=32),
                pop_size=100,
            ),
            dim=40,
            num_cluster=4,
            num_mask=2,
            change_every=10,
        ),
        problem=problems.classic.Ackley(),
        fitness_transform=monitor.update
    )
    # init the pipeline
    key = jax.random.PRNGKey(42)
    state = pipeline.init(key)

    # run the pipeline for 50 steps
    for i in range(50):
        state = pipeline.step(state)

    min_fitness = monitor.get_min_fitness()
    assert abs(min_fitness - 16.529775619506836) < 1e-4
