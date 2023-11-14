from evox import workflows, algorithms, problems
from evox.monitors import StdSOMonitor
import jax
import jax.numpy as jnp
import pytest


def test_distributed_cso():
    monitor = StdSOMonitor()
    # create a workflow
    workflow = workflows.RayDistributedWorkflow(
        algorithm=algorithms.CSO(
            lb=jnp.full(shape=(2,), fill_value=-32),
            ub=jnp.full(shape=(2,), fill_value=32),
            pop_size=20,
        ),
        problem=problems.numerical.Ackley(),
        num_workers=2,
        monitor=monitor,
        options={"num_cpus": 0.5, "num_gpus": 0},  # just for testing purpose
    )
    # init the workflow
    key = jax.random.PRNGKey(42)
    state = workflow.init(key)

    # run the workflow for 100 steps
    for i in range(100):
        state = workflow.step(state)

    # the result should be close to 0
    min_fitness = monitor.get_best_fitness()
    print(min_fitness)
    assert min_fitness < 1e-4
