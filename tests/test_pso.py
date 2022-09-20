import evoxlib as exl
from evoxlib import algorithms, problems, pipelines
import jax
import jax.numpy as jnp
import pytest


def test_pso():
    # create a pipeline
    pipeline = pipelines.StdPipeline(
        algorithm=algorithms.PSO(
            lb=jnp.full(shape=(2,), fill_value=-32),
            ub=jnp.full(shape=(2,), fill_value=32),
            pop_size=100,
            inertia_weight=0.6,
            cognitive_coefficient=1.0,
            social_coefficient=2.0,
        ),
        problem=problems.classic.Ackley(),
        fitness_monitor=True
    )
    # init the pipeline
    key = jax.random.PRNGKey(42)
    state = pipeline.init(key)

    # run the pipeline for 1000 steps
    for i in range(100):
        state = pipeline.step(state)

    # the result should be close to 0
    state, min_fitness = pipeline.get_min_fitness(state)
    assert min_fitness < 1e-4
