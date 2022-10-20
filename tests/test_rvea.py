import evox as ex
from evox import pipelines, algorithms, problems
import jax
import jax.numpy as jnp
import pytest


def test_rvea():
    key = jax.random.PRNGKey(123)
    pipeline = pipelines.StdPipeline(
        algorithm=ex.algorithms.RVEA(
            lb=jnp.full(shape=(3,), fill_value=0),
            ub=jnp.full(shape=(3,), fill_value=1),
            n_objs=3,
            pop_size=105,
        ),
        problem=ex.problems.classic.DTLZ1(m=3),
        fitness_monitor=True
    )
    state = pipeline.init(key)

    for i in range(100):
        state = pipeline.step(state)
