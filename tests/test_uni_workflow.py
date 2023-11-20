from evox import workflows, algorithms, problems
from evox.monitors import StdSOMonitor
import jax
import jax.numpy as jnp


def run_uni_workflow_with_jit_problem():
    monitor = StdSOMonitor()
    # create a workflow
    workflow = workflows.UniWorkflow(
        algorithm=algorithms.CSO(
            lb=jnp.full(shape=(2,), fill_value=-32),
            ub=jnp.full(shape=(2,), fill_value=32),
            pop_size=20,
        ),
        problem=problems.numerical.Ackley(),
        monitor=monitor,
    )
    # init the workflow
    key = jax.random.PRNGKey(42)
    state = workflow.init(key)
    state = workflow.enable_multi_devices(state)

    # run the workflow for 100 steps
    for i in range(100):
        state = workflow.step(state)

    monitor.close()
    min_fitness = monitor.get_best_fitness()
    return min_fitness


def run_uni_workflow_with_non_jit_problem():
    monitor = StdSOMonitor()
    # create a workflow
    workflow = workflows.UniWorkflow(
        algorithm=algorithms.CSO(
            lb=jnp.full(shape=(2,), fill_value=-32),
            ub=jnp.full(shape=(2,), fill_value=32),
            pop_size=20,
        ),
        problem=problems.numerical.Ackley(),
        monitor=monitor,
        jit_problem=False,
    )
    # init the workflow
    key = jax.random.PRNGKey(42)
    state = workflow.init(key)

    # run the workflow for 100 steps
    for i in range(100):
        state = workflow.step(state)

    monitor.close()
    min_fitness = monitor.get_best_fitness()
    return min_fitness


def test_uni_workflow_sanity_check():
    monitor = StdSOMonitor()
    # create a workflow
    workflow = workflows.UniWorkflow(
        algorithm=algorithms.PSO(
            lb=jnp.full(shape=(2,), fill_value=-1),
            ub=jnp.full(shape=(2,), fill_value=1),
            pop_size=20,
        ),
        problem=problems.numerical.Sphere(),
        monitor=monitor,
        jit_problem=True,
    )

    key = jax.random.PRNGKey(42)
    state = workflow.init(key)

    for i in range(10):
        state = workflow.step(state)

    monitor.close()
    min_fitness = monitor.get_best_fitness()
    assert min_fitness < 1e-2


def test_uni_workflow():
    min_fitness2 = run_uni_workflow_with_non_jit_problem()
    min_fitness1 = run_uni_workflow_with_jit_problem()
    assert abs(min_fitness1 - min_fitness2) < 1e-4
    assert min_fitness1 < 1e-4
