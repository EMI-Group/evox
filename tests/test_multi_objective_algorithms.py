from evox import pipelines, algorithms, problems
from evox.monitors import StdMOMonitor
from evox.metrics import IGD
import jax
import jax.numpy as jnp


def run_moea(algorithm, problem=problems.classic.DTLZ1(m=3)):
    key = jax.random.PRNGKey(123)
    monitor = StdMOMonitor(record_pf=False)
    problem = problems.classic.DTLZ1(m=3)
    pipeline = pipelines.StdPipeline(
        algorithm=algorithm,
        problem=problem,
        fitness_transform=monitor.record_fit,
    )
    state = pipeline.init(key)
    true_pf, state = problem.pf(state)

    for i in range(10):
        state = pipeline.step(state)

    objs = monitor.get_last()
    igd = IGD(true_pf, objs).calulate()
    print("igd", igd)


def test_ibea():
    algorithm = algorithms.IBEA(
        lb=jnp.full(shape=(3,), fill_value=0),
        ub=jnp.full(shape=(3,), fill_value=1),
        n_objs=3,
        pop_size=100,
    )
    run_moea(algorithm)


def test_moead():
    algorithm = algorithms.MOEAD(
        lb=jnp.full(shape=(3,), fill_value=0),
        ub=jnp.full(shape=(3,), fill_value=1),
        n_objs=3,
        pop_size=100,
        type=1,
    )
    run_moea(algorithm)


def test_nsga2():
    algorithm = algorithms.NSGA2(
        lb=jnp.full(shape=(3,), fill_value=0),
        ub=jnp.full(shape=(3,), fill_value=1),
        n_objs=3,
        pop_size=100,
    )
    run_moea(algorithm)


def test_rvea():
    algorithm = algorithms.RVEA(
        lb=jnp.full(shape=(3,), fill_value=0),
        ub=jnp.full(shape=(3,), fill_value=1),
        n_objs=3,
        pop_size=100,
    )
    run_moea(algorithm)


def test_nsga3():
    algorithm = algorithms.NSGA3(
        lb=jnp.full(shape=(3,), fill_value=0),
        ub=jnp.full(shape=(3,), fill_value=1),
        n_objs=3,
        pop_size=100,
    )
    run_moea(algorithm)


def test_eagmoead():
    algorithm = algorithms.EAGMOEAD(
        lb=jnp.full(shape=(3,), fill_value=0),
        ub=jnp.full(shape=(3,), fill_value=1),
        n_objs=3,
        pop_size=100,
    )
    run_moea(algorithm)


def test_hype():
    algorithm = algorithms.HypE(
        lb=jnp.full(shape=(3,), fill_value=0),
        ub=jnp.full(shape=(3,), fill_value=1),
        n_objs=3,
        pop_size=100,
    )
    run_moea(algorithm)


def test_moeaddra():
    algorithm = algorithms.MOEADDRA(
        lb=jnp.full(shape=(3,), fill_value=0),
        ub=jnp.full(shape=(3,), fill_value=1),
        n_objs=3,
        pop_size=100,
    )
    run_moea(algorithm)


def test_spea2():
    algorithm = algorithms.SPEA2(
        lb=jnp.full(shape=(3,), fill_value=0),
        ub=jnp.full(shape=(3,), fill_value=1),
        n_objs=3,
        pop_size=100,
    )
    run_moea(algorithm)
