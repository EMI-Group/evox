import evox as ex
import jax
import jax.numpy as jnp
import pytest
from evox import pipelines, problems
from evox.algorithms import CMA_ES, CSO, DE, PGPE, PSO, OpenES, xNES
from evox.monitors import FitnessMonitor
from evox.utils import compose, rank_based_fitness


def run_single_objective_algorithm(
    algorithm, problem=problems.classic.Sphere(), num_iter=200, fitness_shaping=False
):
    key = jax.random.PRNGKey(42)
    monitor = FitnessMonitor(keep_global_best=True)
    if fitness_shaping is True:
        fitness_transform = compose(monitor.update, rank_based_fitness)
    else:
        fitness_transform = monitor.update

    pipeline = pipelines.StdPipeline(
        algorithm=algorithm,
        problem=problem,
        fitness_transform=fitness_transform,
    )

    state = pipeline.init(key)

    for i in range(num_iter):
        state = pipeline.step(state)

    return monitor.get_min_fitness()


def test_cso():
    lb = jnp.full((5,), -32.0)
    ub = jnp.full((5,), 32.0)
    algorithm = CSO(lb, ub, 100)
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
    assert fitness < 0.1


def test_openes():
    init_mean = jnp.array([5.0, -10, 15, -20, 25])

    algorithm = OpenES(
        init_mean, 100, learning_rate=1, noise_std=3, mirrored_sampling=True
    )
    fitness = run_single_objective_algorithm(
        algorithm, fitness_shaping=True, num_iter=1000
    )
    assert fitness < 1


def test_xnes():
    init_mean = jnp.array([5.0, -10, 15, -20, 25])
    init_covar = jnp.eye(5) * 2
    algorithm = xNES(init_mean, init_covar, pop_size=100)
    fitness = run_single_objective_algorithm(algorithm)
    assert fitness < 0.1


def test_de():
    lb = jnp.full((5,), -32.0)
    ub = jnp.full((5,), 32.0)
    algorithm = DE(lb, ub, 100, batch_size=100, base_vector="rand")
    fitness = run_single_objective_algorithm(algorithm)
    assert fitness < 0.1
