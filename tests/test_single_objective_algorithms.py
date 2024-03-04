import jax
import jax.numpy as jnp
import pytest
from evox import workflows, problems
from evox.algorithms import (
    CMAES,
    SepCMAES,
    CPSOS,
    CSO,
    DE,
    PGPE,
    PSO,
    OpenES,
    XNES,
    CoDE,
    JaDE,
    SaDE,
    SHADE,
    ODE,
    create_batch_algorithm,
    decoder_de,
    MetaDE,
    ParamDE,
)
from evox.monitors import StdSOMonitor
from evox.utils import compose, rank_based_fitness


def run_single_objective_algorithm(
    algorithm, problem=problems.numerical.Sphere(), num_iter=200, fitness_shaping=False
):
    key = jax.random.PRNGKey(42)
    monitor = StdSOMonitor()
    if fitness_shaping is True:
        fit_transforms = [rank_based_fitness]
    else:
        fit_transforms = []

    workflow = workflows.StdWorkflow(
        algorithm=algorithm,
        problem=problem,
        monitors=[monitor],
        fit_transforms=fit_transforms,
    )

    state = workflow.init(key)

    for i in range(num_iter):
        state = workflow.step(state)

    return monitor.get_best_fitness()


def test_cpso_s():
    lb = jnp.full((5,), -32.0)
    ub = jnp.full((5,), 32.0)
    algorithm = CPSOS(
        lb,
        ub,
        100,
        inertia_weight=0.6,
        cognitive_coefficient=2.5,
        social_coefficient=0.8,
    )
    fitness = run_single_objective_algorithm(algorithm)
    assert fitness < 0.1


def test_cso():
    lb = jnp.full((5,), -32.0)
    ub = jnp.full((5,), 32.0)
    algorithm = CSO(lb, ub, 100)
    fitness = run_single_objective_algorithm(algorithm)
    assert fitness < 0.1


def test_cma_es():
    init_mean = jnp.array([5.0, -10, 15, -20, 25])
    algorithm = CMAES(init_mean, init_stdev=0.1, pop_size=10)
    fitness = run_single_objective_algorithm(algorithm)
    assert fitness < 0.1


def test_sep_cma_es():
    init_mean = jnp.array([5.0, -10, 15, -20, 25])
    algorithm = SepCMAES(init_mean, init_stdev=0.1, pop_size=10)
    fitness = run_single_objective_algorithm(algorithm)
    assert fitness < 0.1


def test_pso():
    lb = jnp.full((5,), -32.0)
    ub = jnp.full((5,), 32.0)
    algorithm = PSO(lb, ub, 100)
    fitness = run_single_objective_algorithm(algorithm)
    assert fitness < 0.1


@pytest.mark.parametrize("optimizer", ["adam", "clipup"])
def test_pgpe(optimizer):
    init_mean = jnp.array([5.0, -10, 15, -20, 25])
    algorithm = PGPE(
        100,
        init_mean,
        optimizer=optimizer,
        center_learning_rate=0.3,
        stdev_init=10,
        stdev_learning_rate=0.2,
    )
    fitness = run_single_objective_algorithm(algorithm, fitness_shaping=True)
    print(fitness)
    assert fitness < 0.1


@pytest.mark.parametrize("optimizer", ["adam", None])
def test_openes(optimizer):
    init_mean = jnp.array([5.0, -10, 15, -20, 25])

    algorithm = OpenES(
        init_mean,
        100,
        learning_rate=1,
        noise_stdev=3,
        optimizer=optimizer,
        mirrored_sampling=True,
    )
    fitness = run_single_objective_algorithm(
        algorithm, fitness_shaping=True, num_iter=1000
    )
    assert fitness < 1


def test_xnes():
    init_mean = jnp.array([5.0, -10, 15, -20, 25])
    init_covar = jnp.eye(5) * 2
    algorithm = XNES(init_mean, init_covar, pop_size=100)
    fitness = run_single_objective_algorithm(algorithm)
    assert fitness < 0.1


def test_de():
    lb = jnp.full((5,), -32.0)
    ub = jnp.full((5,), 32.0)
    algorithm = DE(lb, ub, 100, batch_size=100, base_vector="rand")
    fitness = run_single_objective_algorithm(algorithm)
    assert fitness < 0.1


def test_ode():
    lb = jnp.full((5,), -32.0)
    ub = jnp.full((5,), 32.0)
    algorithm = ODE(
        lb=lb,
        ub=ub,
        pop_size=100,
    )
    fitness = run_single_objective_algorithm(algorithm, num_iter=60)
    # A real step actually calls the tell function twice, so num_iter is actually 30
    assert fitness < 0.1


def test_code():
    lb = jnp.full((5,), -32.0)
    ub = jnp.full((5,), 32.0)
    algorithm = CoDE(lb, ub, pop_size=100)
    fitness = run_single_objective_algorithm(algorithm, num_iter=30)
    assert fitness < 0.1


def test_jade():
    lb = jnp.full((5,), -32.0)
    ub = jnp.full((5,), 32.0)
    algorithm = JaDE(lb, ub, pop_size=1000)
    fitness = run_single_objective_algorithm(algorithm, num_iter=30)
    assert fitness < 0.1


def test_sade():
    lb = jnp.full((5,), -32.0)
    ub = jnp.full((5,), 32.0)
    algorithm = SaDE(lb, ub, pop_size=100)
    fitness = run_single_objective_algorithm(algorithm, num_iter=30)
    assert fitness < 0.1


def test_shade():
    lb = jnp.full((5,), -32.0)
    ub = jnp.full((5,), 32.0)
    algorithm = SHADE(lb, ub, pop_size=100)
    fitness = run_single_objective_algorithm(algorithm, num_iter=30)
    assert fitness < 0.1


def test_metade():
    # MetaDE inherits from the problem class and has a different computational flow from the above algorithms
    tiny_num = 1e-5
    param_lb = jnp.array([0, 0, 0, 0, 1, 0])
    param_ub = jnp.array([1, 1, 4 - tiny_num, 4 - tiny_num, 5 - tiny_num, 3 - tiny_num])
    algorithm = DE(
        lb=param_lb,
        ub=param_ub,
        pop_size=100,
        base_vector="rand",
        differential_weight=0.5,
        cross_probability=0.9,
    )
    BatchDE = create_batch_algorithm(base_algorithm=ParamDE, batch_size=100, num_runs=1)
    batch_de = BatchDE(lb=jnp.full((5,), -32.0), ub=jnp.full((5,), 32.0), pop_size=100)
    base_problem = problems.numerical.Sphere()
    decoder = decoder_de
    key = jax.random.PRNGKey(42)
    monitor = StdSOMonitor()
    meta_problem = MetaDE(
        batch_de, base_problem, batch_size=100, num_runs=1, base_alg_steps=100
    )

    workflow = workflows.StdWorkflow(
        algorithm=algorithm,
        problem=meta_problem,
        pop_transform=decoder,
        monitor=monitor,
    )
    key, _ = jax.random.split(key)
    state = workflow.init(key)

    power_up = 0
    for i in range(30):
        if i == 30 - 1:
            power_up = 1
        state = state.update_child("problem", {"power_up": power_up})
        state = workflow.step(state)
    assert monitor.get_best_fitness() < 0.1
