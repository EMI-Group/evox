from evox import pipelines, algorithms, problems
from evox.monitors import StdSOMonitor
import jax
import jax.numpy as jnp
import pytest


def test_distributed_cso():
    monitor = StdSOMonitor()
    # create a pipeline
    pipeline = pipelines.RayDistributedWorkflow(
        algorithm=algorithms.CSO(
            lb=jnp.full(shape=(2,), fill_value=-32),
            ub=jnp.full(shape=(2,), fill_value=32),
            pop_size=20,
        ),
        problem=problems.numerical.Ackley(),
        pop_size=10,
        num_workers=2,
        global_fitness_transform=monitor.record_fit,
        options={"num_cpus": 0.5, "num_gpus": 0},  # just for testing purpose
    )
    # init the pipeline
    key = jax.random.PRNGKey(42)
    state = pipeline.init(key)

    # run the pipeline for 100 steps
    for i in range(100):
        state = pipeline.step(state)

    # the result should be close to 0
    min_fitness = monitor.get_min_fitness()
    print(min_fitness)
    assert min_fitness < 1e-4
    pipeline.health_check(state)
