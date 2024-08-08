import jax
import jax.numpy as jnp
import pytest

from evox import State, algorithms, problems, workflows
from evox.monitors import EvalMonitor, PopMonitor, StdMOMonitor, StdSOMonitor


def test_std_so_monitor():
    monitor = StdSOMonitor(record_topk=1, record_fit_history=True)

    pop1 = jnp.arange(15).reshape((3, 5))
    fitness1 = jnp.arange(3)
    monitor.record_pop(pop1)
    monitor.record_fit(fitness1)
    assert monitor.get_best_fitness() == 0
    assert monitor.get_topk_fitness() == 0
    assert (monitor.get_best_solution() == pop1[0]).all()
    assert (monitor.get_topk_solutions() == pop1[0:1]).all()

    pop2 = -jnp.arange(15).reshape((3, 5))
    fitness2 = -jnp.arange(3)
    monitor.record_pop(pop2)
    monitor.record_fit(fitness2)
    assert monitor.get_best_fitness() == -2
    assert monitor.get_topk_fitness() == -2
    assert (monitor.get_best_solution() == pop2[2]).all()
    assert (monitor.get_topk_solutions() == pop2[2:3]).all()


def test_std_mo_monitor():
    monitor = StdMOMonitor(record_pf=True, record_fit_history=True)

    pop1 = jnp.arange(15).reshape((3, 5))
    fitness1 = jnp.array([[1, 2], [3, 1], [5, 6]])
    monitor.record_pop(pop1)
    monitor.record_fit(fitness1)
    assert (monitor.get_pf_fitness() == jnp.array([[1, 2], [3, 1]])).all()
    assert (monitor.get_pf_solutions() == pop1[jnp.array([0, 1])]).all()

    pop2 = -jnp.arange(15).reshape((3, 5))
    fitness2 = jnp.array([[0.5, 1.5], [7, 8], [0, 10]])
    monitor.record_pop(pop2)
    monitor.record_fit(fitness2)
    assert (monitor.get_pf_fitness() == jnp.array([[3, 1], [0.5, 1.5], [0, 10]])).all()
    assert (
        monitor.get_pf_solutions()
        == jnp.concatenate([pop1[jnp.array([1])], pop2[jnp.array([0, 2])]], axis=0)
    ).all()


@pytest.mark.parametrize(
    "full_fit_history,full_sol_history,topk",
    [
        (False, False, 1),
        (False, False, 2),
        (False, True, 1),
        (False, True, 2),
        (True, False, 1),
        (True, False, 2),
        (True, False, 1),
        (True, True, 2),
    ],
)
def test_eval_monitor_with_so(full_fit_history, full_sol_history, topk):
    monitor = EvalMonitor(
        full_fit_history=full_fit_history, full_sol_history=full_sol_history, topk=topk
    )
    monitor.set_opt_direction = 1

    pop1 = jnp.arange(15).reshape((3, 5))
    fitness1 = jnp.arange(3)

    state = State()

    state = monitor.post_eval(state, pop1, None, fitness1)
    state = state.execute_callbacks()
    assert monitor.get_best_fitness() == 0
    assert (monitor.get_topk_fitness() == fitness1[:topk]).all()
    assert (monitor.get_best_solution() == pop1[0]).all()
    assert (monitor.get_topk_solutions() == pop1[:topk]).all()

    pop2 = -jnp.arange(15).reshape((3, 5))
    fitness2 = -jnp.arange(3)
    state = monitor.post_eval(state, pop2, None, fitness2)
    state = state.execute_callbacks()
    assert monitor.get_best_fitness() == -2
    assert (monitor.get_topk_fitness() == fitness2[-topk:][::-1]).all()
    assert (monitor.get_best_solution() == pop2[-1]).all()
    assert (monitor.get_topk_solutions() == pop2[-topk:][::-1]).all()


@pytest.mark.parametrize(
    "full_fit_history,full_sol_history",
    [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ],
)
def test_eval_monitor_with_mo(full_fit_history, full_sol_history):
    monitor = EvalMonitor(
        full_fit_history=full_fit_history, full_sol_history=full_sol_history
    )
    monitor.set_opt_direction = 1

    pop1 = jnp.arange(15).reshape((3, 5))
    fitness1 = jnp.arange(6).reshape(3, 2)

    state = State()

    state = monitor.post_eval(state, pop1, None, fitness1)
    state = state.execute_callbacks()
    assert (monitor.get_latest_fitness() == fitness1).all()
    assert (monitor.get_latest_solution() == pop1).all()

    pop2 = -jnp.arange(15).reshape((3, 5))
    fitness2 = -jnp.arange(6).reshape(3, 2)
    state = monitor.post_eval(state, pop2, None, fitness2)
    state = state.execute_callbacks()
    assert (monitor.get_latest_fitness() == fitness2).all()
    assert (monitor.get_latest_solution() == pop2).all()


@pytest.mark.parametrize("fitness_only", [True, False])
def test_pop_monitor(fitness_only):
    monitor = PopMonitor(fitness_only=fitness_only)
    algorithm = algorithms.CSO(lb=jnp.zeros((5,)), ub=jnp.ones((5,)), pop_size=4)
    problem = problems.numerical.Sphere()
    workflow = workflows.StdWorkflow(algorithm, problem, monitors=[monitor])
    key = jax.random.PRNGKey(0)
    state = workflow.init(key)
    state = workflow.step(state)
    assert (
        monitor.get_latest_fitness() == state.get_child_state("algorithm").fitness
    ).all()
    if not fitness_only:
        assert (
            monitor.get_latest_population()
            == state.get_child_state("algorithm").population
        ).all()
