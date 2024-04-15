import jax
import jax.numpy as jnp
import pytest
from evox import algorithms, workflows, problems, Stateful
from evox.monitors import EvalMonitor


@pytest.mark.skip(
    reason="a bit non-deterministic now, maybe due to the fact that eigen decomposition is unstable"
)
def test_clustered_cma_es():
    # create a workflow
    init_mean = jnp.full((10,), fill_value=-20)
    monitor = EvalMonitor()
    workflow = workflows.StdWorkflow(
        algorithms.ClusterdAlgorithm(
            base_algorithm=algorithms.CMAES(init_mean, init_stdev=10, pop_size=10),
            dim=40,
            num_cluster=4,
        ),
        problem=problems.numerical.Ackley(),
        monitors=[monitor],
    )
    # init the workflow
    key = jax.random.PRNGKey(42)
    state = workflow.init(key)

    # run the workflow for 10 steps
    for i in range(200):
        state = workflow.step(state)

    min_fitness = monitor.get_best_fitness()
    assert min_fitness < 2


@pytest.mark.parametrize("random_subpop", [True, False])
def test_vectorized_coevolution(random_subpop):
    # create a workflow
    monitor = EvalMonitor()
    base_algorithm = algorithms.CSO(
        lb=jnp.full(shape=(20,), fill_value=-32),
        ub=jnp.full(shape=(20,), fill_value=32),
        pop_size=100,
    )
    base_algorithms = Stateful.stack([base_algorithm] * 2)
    algorithm = algorithms.VectorizedCoevolution(
        base_algorithms=base_algorithms,
        dim=40,
        num_subpops=2,
        random_subpop=random_subpop,
    )
    workflow = workflows.StdWorkflow(
        algorithm,
        problem=problems.numerical.Ackley(),
        monitors=[monitor],
    )
    # init the workflow
    key = jax.random.PRNGKey(42)
    state = workflow.init(key)

    for i in range(200):
        state = workflow.step(state)
    
    monitor.close()

    min_fitness = monitor.get_best_fitness()
    assert min_fitness < 0.5


@pytest.mark.parametrize("random_subpop", [True, False])
def test_coevolution(random_subpop):
    # create a workflow
    monitor = EvalMonitor()
    base_algorithm = algorithms.CSO(
        lb=jnp.full(shape=(20,), fill_value=-32),
        ub=jnp.full(shape=(20,), fill_value=32),
        pop_size=100,
    )
    base_algorithms = Stateful.stack([base_algorithm] * 2)
    algorithm = algorithms.Coevolution(
        base_algorithms,
        dim=40,
        num_subpops=2,
        random_subpop=random_subpop,
    )

    workflow = workflows.StdWorkflow(
        algorithm=algorithm,
        problem=problems.numerical.Ackley(),
        monitors=[monitor],
    )
    # init the workflow
    key = jax.random.PRNGKey(42)
    state = workflow.init(key)
    for i in range(400):
        state = workflow.step(state)
    
    monitor.close()

    min_fitness = monitor.get_best_fitness()
    assert min_fitness < 0.5


@pytest.mark.skip(reason="currently random_mask is unstable")
def test_random_mask_cso():
    # create a workflow
    monitor = EvalMonitor()
    workflow = workflows.StdWorkflow(
        algorithms.RandomMaskAlgorithm(
            base_algorithm=algorithms.CSO(
                lb=jnp.full(shape=(10,), fill_value=-32),
                ub=jnp.full(shape=(10,), fill_value=32),
                pop_size=100,
            ),
            dim=40,
            num_cluster=4,
            num_mask=2,
            change_every=10,
            pop_size=50,
        ),
        problem=problems.numerical.Ackley(),
        monitors=[monitor],
    )
    # init the workflow
    key = jax.random.PRNGKey(42)
    state = workflow.init(key)

    # run the workflow for 10 steps
    for i in range(10):
        state = workflow.step(state)

    min_fitness = monitor.get_best_fitness()
    print(min_fitness)
    assert abs(min_fitness - 19.6) < 0.1
