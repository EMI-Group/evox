import evox as ex
from evox import pipelines, algorithms, problems
import jax
import jax.numpy as jnp
import pytest


def test_moead():
    key = jax.random.PRNGKey(123)
    pipeline = pipelines.StdPipeline(
        algorithm=ex.algorithms.MOEAD(
            lb=jnp.full(shape=(2,), fill_value=0),
            ub=jnp.full(shape=(2,), fill_value=1),
            n_objs=2,
            pop_size=100,
            type=1,
        ),
        problem=ex.problems.classic.ZDT1(n=2),
        fitness_monitor=True
    )
    state = pipeline.init(key)

    for i in range(100):
        state = pipeline.step(state)
