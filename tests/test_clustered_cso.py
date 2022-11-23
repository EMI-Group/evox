import evox as ex
import jax
import jax.numpy as jnp
import pytest
from evox import algorithms, pipelines, problems
from evox.monitors import FitnessMonitor
import chex


@pytest.mark.parametrize("num_gpus", [None, 1])
def test_clustered_cma_es(num_gpus):
    # create a pipeline
    init_mean = jnp.full((10,), fill_value=-20)
    monitor = FitnessMonitor()
    pipeline = pipelines.StdPipeline(
        algorithms.ClusterdAlgorithm(
            base_algorithm=ex.algorithms.CMA_ES(init_mean, init_stdvar=10, pop_size=10),
            dim=40,
            num_cluster=4,
            num_gpus=num_gpus,
        ),
        problem=problems.classic.Ackley(),
        fitness_transform=monitor.update,
    )
    # init the pipeline
    key = jax.random.PRNGKey(42)
    state = pipeline.init(key)

    # run the pipeline for 10 steps
    for i in range(10):
        state = pipeline.step(state)

    min_fitness = monitor.get_min_fitness()
    print(min_fitness)
    assert abs(min_fitness - 21) < 0.1


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
        fitness_transform=monitor.update,
    )
    # init the pipeline
    key = jax.random.PRNGKey(42)
    state = pipeline.init(key)

    # run the pipeline for 10 steps
    for i in range(10):
        state = pipeline.step(state)

    min_fitness = monitor.get_min_fitness()
    assert abs(min_fitness - 19.6) < 0.1
