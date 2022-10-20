from evox import pipelines, algorithms, problems
import jax
import jax.numpy as jnp
import pytest


def test_moead():
    key = jax.random.PRNGKey(123)
    pipeline = pipelines.StdPipeline(
        algorithm=algorithms.MOEAD(
            lb=jnp.full(shape=(2,), fill_value=0),
            ub=jnp.full(shape=(2,), fill_value=1),
            n_objs=2,
            pop_size=100,
            type=1,
        ),
        problem=problems.classic.ZDT1(n=2),
    )
    state = pipeline.init(key)

    for i in range(2):
        state = pipeline.step(state)
