import evox as ex
from evox import pipelines, algorithms, problems
from evox.monitors import FitnessMonitor
import jax
import jax.numpy as jnp
import pytest


def test_nsga2():
    monitor = FitnessMonitor()
    key = jax.random.PRNGKey(123)
    pipeline = pipelines.StdPipeline(
        algorithm=ex.algorithms.NSGA2(
            lb=jnp.full(shape=(2,), fill_value=0),
            ub=jnp.full(shape=(2,), fill_value=1),
            n_objs=2,
            pop_size=100,
        ),
        problem=ex.problems.classic.ZDT1(n=2),
        fitness_transform=monitor.update
    )
    state = pipeline.init(key)

    for i in range(100):
        state = pipeline.step(state)
