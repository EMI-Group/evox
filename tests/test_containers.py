import evox as ex
import jax
import jax.numpy as jnp
import pytest
from evox import algorithms, pipelines, problems
from evox.monitors import FitnessMonitor
import chex


def test_clustered_cma_es():
    # create a pipeline
    init_mean = jnp.full((10,), fill_value=-20)
    monitor = FitnessMonitor()
    pipeline = pipelines.StdPipeline(
        algorithms.ClusterdAlgorithm(
            base_algorithm=ex.algorithms.CMAES(init_mean, init_stdev=10, pop_size=10),
            dim=40,
            num_cluster=4,
        ),
        problem=problems.classic.Ackley(),
        fitness_transform=monitor.update,
    )
    # init the pipeline
    key = jax.random.PRNGKey(42)
    state = pipeline.init(key)

    # run the pipeline for 10 steps
    for i in range(200):
        state = pipeline.step(state)

    min_fitness = monitor.get_min_fitness()
    assert min_fitness < 1


@pytest.mark.parametrize("random_subpop", [True, False])
def test_vectorized_coevolution(random_subpop):
    # create a pipeline
    monitor = FitnessMonitor()
    pipeline = pipelines.StdPipeline(
        algorithms.VectorizedCoevolution(
            base_algorithm=algorithms.CSO(
                lb=jnp.full(shape=(10,), fill_value=-32),
                ub=jnp.full(shape=(10,), fill_value=32),
                pop_size=20,
            ),
            dim=40,
            num_subpops=4,
            random_subpop=random_subpop,
        ),
        problem=problems.classic.Ackley(),
        fitness_transform=monitor.update,
    )
    # init the pipeline
    key = jax.random.PRNGKey(42)
    state = pipeline.init(key)

    for i in range(200):
        state = pipeline.step(state)

    min_fitness = monitor.get_min_fitness()
    assert min_fitness < 0.1


@pytest.mark.parametrize("random_subpop", [True, False])
def test_coevolution(random_subpop):
    # create a pipeline
    monitor = FitnessMonitor()
    pipeline = pipelines.StdPipeline(
        algorithms.Coevolution(
            base_algorithm=algorithms.CSO(
                lb=jnp.full(shape=(10,), fill_value=-32),
                ub=jnp.full(shape=(10,), fill_value=32),
                pop_size=20,
            ),
            dim=40,
            num_subpops=4,
            subpop_size=10,
            random_subpop=random_subpop,
        ),
        problem=problems.classic.Ackley(),
        fitness_transform=monitor.update,
    )
    # init the pipeline
    key = jax.random.PRNGKey(42)
    state = pipeline.init(key)

    for i in range(4 * 200):
        state = pipeline.step(state)

    min_fitness = monitor.get_min_fitness()
    assert min_fitness < 0.5


@pytest.mark.skip(reason="currently random_mask is unstable")
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
            pop_size=50
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
    assert abs(min_fitness - 19.6) < 0.1
