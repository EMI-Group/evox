import evoxlib as exl
from evoxlib import pipelines, algorithms, problems
import jax
import jax.numpy as jnp
import pytest


def test_nsga2():
    key = jax.random.PRNGKey(123)
    pipeline = pipelines.StdPipeline(
        algorithm=exl.algorithms.NSGA2(
            lb=jnp.full(shape=(2,), fill_value=0),
            ub=jnp.full(shape=(2,), fill_value=1),
            n_objs=2,
            pop_size=100,
        ),
        problem=exl.problems.classic.ZDT1(n=2),
        fitness_monitor=True
    )
    state = pipeline.init(key)

    for i in range(100):
        state = pipeline.step(state)
