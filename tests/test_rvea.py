import sys
sys.path.append("../src")

from evox import pipelines, algorithms, problems
from evox.monitors import FitnessMonitor
from evox.metrics import IGD
import jax
import jax.numpy as jnp
import pytest, time


def test_rvea():
    key = jax.random.PRNGKey(123)
    problem = problems.classic.DTLZ2(m=3)
    monitor = FitnessMonitor(n_objects=3)
    pipeline = pipelines.StdPipeline(
        algorithm=algorithms.RVEA(
            lb=jnp.full(shape=(3,), fill_value=0),
            ub=jnp.full(shape=(3,), fill_value=1),
            n_objs=3,
            pop_size=100,
        ),
        problem=problem,
        fitness_transform=monitor.update
    )
    state = pipeline.init(key)
    pf, state = problem.pf(state=state)


    for i in range(100):
        state = pipeline.step(state)
        objs = monitor.get_last()
        igd = IGD(pf, objs).calulate()
        print("step", i)
        print("igd", igd)
