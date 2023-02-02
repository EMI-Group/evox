import sys
sys.path.append("../src")

from evox import pipelines, algorithms, problems
from evox.monitors import FitnessMonitor
from evox.metrics import HyperVolume, IGD
import jax
import jax.numpy as jnp
import pytest


def test_nsga2():
    monitor = FitnessMonitor(n_objects=3)
    key = jax.random.PRNGKey(1234)
    problem=problems.classic.DTLZ1(m=3)
    pipeline = pipelines.StdPipeline(
        algorithm=algorithms.NSGA2(
            lb=jnp.full(shape=(3,), fill_value=0),
            ub=jnp.full(shape=(3,), fill_value=1),
            n_objs=3,
            pop_size=100,
        ),
        # problem=ex.problems.classic.ZDT1(n=2),
        problem=problem,
        fitness_transform=monitor.update
    )
    state = pipeline.init(key)
    pf, state = problem.pf(state=state)

    for i in range(100):
        state = pipeline.step(state)

        objs = monitor.get_last()
        # print(objs)
        igd = IGD(pf, objs).calulate()
        print("step", i)
        # print(t1-t0)
        print("igd", igd)

