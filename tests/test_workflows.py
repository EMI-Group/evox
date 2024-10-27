import jax
import jax.numpy as jnp
import pytest

from evox import algorithms, problems, use_state, workflows
from evox.monitors import EvalMonitor


def run_std_workflow_with_jit_problem():
    monitor = EvalMonitor()
    # create a workflow
    workflow = workflows.StdWorkflow(
        algorithm=algorithms.CSO(
            lb=jnp.full(shape=(2,), fill_value=-32),
            ub=jnp.full(shape=(2,), fill_value=32),
            pop_size=20,
        ),
        problem=problems.numerical.Ackley(),
        monitors=[monitor],
    )
    # init the workflow
    key = jax.random.PRNGKey(42)
    state = workflow.init(key)

    # run the workflow for 100 steps
    for i in range(100):
        state = workflow.step(state)

    min_fitness, state = workflow.call_monitor(state, monitor.get_best_fitness)
    return min_fitness


@pytest.mark.skip(reason="non jit with jit-able problem is problematic.")
def run_std_workflow_with_non_jit_problem():
    monitor = EvalMonitor()
    # create a workflow
    workflow = workflows.StdWorkflow(
        algorithm=algorithms.CSO(
            lb=jnp.full(shape=(2,), fill_value=-32),
            ub=jnp.full(shape=(2,), fill_value=32),
            pop_size=20,
        ),
        problem=problems.numerical.Ackley(),
        monitors=[monitor],
        external_problem=True,
        num_objectives=1,
    )
    # init the workflow
    key = jax.random.PRNGKey(42)
    state = workflow.init(key)

    # run the workflow for 100 steps
    for i in range(100):
        state = workflow.step(state)

    min_fitness, state = workflow.call_monitor(state, monitor.get_best_fitness)
    return min_fitness


def test_std_workflow_sanity_check():
    monitor = EvalMonitor()
    # create a workflow
    workflow = workflows.StdWorkflow(
        algorithm=algorithms.PSO(
            lb=jnp.full(shape=(2,), fill_value=-1),
            ub=jnp.full(shape=(2,), fill_value=1),
            pop_size=20,
        ),
        problem=problems.numerical.Sphere(),
        monitors=[monitor],
        external_problem=True,
        num_objectives=1,
    )

    key = jax.random.PRNGKey(42)
    state = workflow.init(key)

    for i in range(10):
        state = workflow.step(state)

    min_fitness, state = workflow.call_monitor(state, monitor.get_best_fitness)
    assert min_fitness < 1e-2


def test_std_workflow():
    min_fitness1 = run_std_workflow_with_jit_problem()
    assert min_fitness1 < 1e-4


@pytest.mark.skip(reason="ray integration is not complete.")
def test_distributed_cso():
    monitor = EvalMonitor()
    # create a workflow
    workflow = workflows.RayDistributedWorkflow(
        algorithm=algorithms.CSO(
            lb=jnp.full(shape=(2,), fill_value=-32),
            ub=jnp.full(shape=(2,), fill_value=32),
            pop_size=20,
        ),
        problem=problems.numerical.Ackley(),
        num_workers=2,
        monitors=[monitor],
        options={"num_cpus": 0.5, "num_gpus": 0},  # just for testing purpose
    )
    # init the workflow
    key = jax.random.PRNGKey(42)
    state = workflow.init(key)

    # run the workflow for 100 steps
    for i in range(100):
        state = workflow.step(state)

    # the result should be close to 0
    min_fitness, state = use_state(monitor.get_best_fitness)(state)
    assert min_fitness < 1e-4
