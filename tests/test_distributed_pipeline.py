import evoxlib as exl
from evoxlib import pipelines, algorithms, problems
from evoxlib.monitors import FitnessMonitor
import jax
import jax.numpy as jnp
import pytest


def test_cso():
    monitor = FitnessMonitor()
    # create a pipeline
    pipeline = pipelines.DistributedPipeline(
        algorithm=algorithms.CSO(
            lb=jnp.full(shape=(2,), fill_value=-32),
            ub=jnp.full(shape=(2,), fill_value=32),
            pop_size=200,
        ),
        problem=problems.classic.Ackley(),
        pop_size=200,
        num_workers=2,
        global_fitness_transform=monitor.update,
        options={
            "num_cpus": 2,
            "num_gpus": 0
        }
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
