from evox import pipelines, algorithms, problems
from evox.monitors import StdSOMonitor
import jax
import jax.numpy as jnp


def run_uni_workflow_with_jit_problem():
    monitor = StdSOMonitor()
    # create a pipeline
    pipeline = pipelines.UniWorkflow(
        algorithm=algorithms.CSO(
            lb=jnp.full(shape=(2,), fill_value=-32),
            ub=jnp.full(shape=(2,), fill_value=32),
            pop_size=20,
        ),
        problem=problems.numerical.Ackley(),
        monitor=monitor,
    )
    # init the pipeline
    key = jax.random.PRNGKey(42)
    state = pipeline.init(key)
    state = pipeline.enable_multi_devices(state)

    # run the pipeline for 100 steps
    for i in range(100):
        state = pipeline.step(state)

    monitor.close()
    min_fitness = monitor.get_min_fitness()
    return min_fitness


def run_uni_workflow_with_non_jit_problem():
    monitor = StdSOMonitor()
    # create a pipeline
    pipeline = pipelines.UniWorkflow(
        algorithm=algorithms.CSO(
            lb=jnp.full(shape=(2,), fill_value=-32),
            ub=jnp.full(shape=(2,), fill_value=32),
            pop_size=20,
        ),
        problem=problems.numerical.Ackley(),
        monitor=monitor,
        jit_problem=False,
    )
    # init the pipeline
    key = jax.random.PRNGKey(42)
    state = pipeline.init(key)

    # run the pipeline for 100 steps
    for i in range(100):
        state = pipeline.step(state)

    monitor.close()
    min_fitness = monitor.get_min_fitness()
    return min_fitness


def test_uni_workflow():
    min_fitness2 = run_uni_workflow_with_non_jit_problem()
    min_fitness1 = run_uni_workflow_with_jit_problem()
    assert abs(min_fitness1 - min_fitness2) < 1e-4
    assert min_fitness1 < 1e-4
