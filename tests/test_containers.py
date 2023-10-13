import jax
import jax.numpy as jnp
import pytest
from evox import algorithms, workflows, problems
from evox.monitors import StdSOMonitor


@pytest.mark.skip(
    reason="a bit non-deterministic now, maybe due to the fact that eigen decomposition is unstable"
)
def test_clustered_cma_es():
    # create a workflow
    init_mean = jnp.full((10,), fill_value=-20)
    monitor = StdSOMonitor()
    workflow = workflows.StdWorkflow(
        algorithms.ClusterdAlgorithm(
            base_algorithm=algorithms.CMAES(init_mean, init_stdev=10, pop_size=10),
            dim=40,
            num_cluster=4,
        ),
        problem=problems.numerical.Ackley(),
        monitor=monitor,
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
    monitor = StdSOMonitor()
    algorithm = algorithms.VectorizedCoevolution(
        base_algorithm=algorithms.CSO(
            lb=jnp.full(shape=(20,), fill_value=-32),
            ub=jnp.full(shape=(20,), fill_value=32),
            pop_size=30,
        ),
        dim=40,
        num_subpops=2,
        random_subpop=random_subpop,
    )
    workflow = workflows.StdWorkflow(
        algorithm,
        problem=problems.numerical.Ackley(),
        monitor=monitor,
    )
    # init the workflow
    key = jax.random.PRNGKey(42)
    state = workflow.init(key)

    if not random_subpop:
        # test the population given by VectorizedCoevolution
        # is the same as manually concatenate the population
        # given by two base algorithms.
        cso1 = algorithms.CSO(
            lb=jnp.full(shape=(20,), fill_value=-32),
            ub=jnp.full(shape=(20,), fill_value=32),
            pop_size=30,
        )
        cso2 = algorithms.CSO(
            lb=jnp.full(shape=(20,), fill_value=-32),
            ub=jnp.full(shape=(20,), fill_value=32),
            pop_size=30,
        )
        _, alg_key = jax.random.split(key)
        key1, key2 = jax.random.split(alg_key)
        cso1_state = cso1.init(key1)
        cso2_state = cso2.init(key2)

        cso1_subpop, _ = cso1.ask(cso1_state)
        cso2_subpop, _ = cso2.ask(cso2_state)
        vcc_cso_pop, _ = algorithm.ask(state)
        pop_size = cso1_subpop.shape[0]
        cso1_pop = jnp.concatenate(
            [cso1_subpop, jnp.tile(cso2_subpop[0, :], (pop_size, 1))], axis=1
        )
        cso2_pop = jnp.concatenate(
            [jnp.tile(cso1_subpop[0, :], (pop_size, 1)), cso2_subpop], axis=1
        )
        target_pop = jnp.concatenate([cso1_pop, cso2_pop], axis=0)
        assert (jnp.abs(vcc_cso_pop - target_pop) < 1e-4).all()

    for i in range(200):
        state = workflow.step(state)

    min_fitness = monitor.get_best_fitness()
    assert min_fitness < 1


@pytest.mark.parametrize(
    "random_subpop, num_subpop_iter", [(True, 1), (False, 1), (True, 2), (False, 2)]
)
def test_coevolution(random_subpop, num_subpop_iter):
    # create a workflow
    monitor = StdSOMonitor()
    workflow = workflows.StdWorkflow(
        algorithms.Coevolution(
            base_algorithm=algorithms.CSO(
                lb=jnp.full(shape=(10,), fill_value=-32),
                ub=jnp.full(shape=(10,), fill_value=32),
                pop_size=20,
            ),
            dim=40,
            num_subpops=4,
            subpop_size=10,
            num_subpop_iter=num_subpop_iter,
            random_subpop=random_subpop,
        ),
        problem=problems.numerical.Ackley(),
        monitor=monitor,
    )
    # init the workflow
    key = jax.random.PRNGKey(42)
    state = workflow.init(key)

    for i in range(4 * 200):
        state = workflow.step(state)

    min_fitness = monitor.get_best_fitness()
    assert min_fitness < 2


@pytest.mark.skip(reason="currently random_mask is unstable")
def test_random_mask_cso():
    # create a workflow
    monitor = StdSOMonitor()
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
        monitor=monitor,
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
