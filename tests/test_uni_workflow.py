import evox as ex
from evox import pipelines, algorithms, problems
from evox.monitors import StdSOMonitor
import jax
import jax.numpy as jnp
import pytest


def test_distributed_cso():
    monitor = StdSOMonitor()
    # create a pipeline
    pipeline = pipelines.UniWorkflow(
        algorithm=algorithms.CSO(
            lb=jnp.full(shape=(2,), fill_value=-32),
            ub=jnp.full(shape=(2,), fill_value=32),
            pop_size=20,
        ),
        problem=problems.classic.Ackley(),
        monitor=monitor
    )
    # init the pipeline
    key = jax.random.PRNGKey(42)
    state = pipeline.init(key)
    state = pipeline.enable_multi_devices(state)

    # run the pipeline for 100 steps
    for i in range(100):
        state = pipeline.step(state)

    # the result should be close to 0
    monitor.close()
    min_fitness = monitor.get_min_fitness()
    print(min_fitness)
    assert min_fitness < 1e-4
