import jax
import jax.numpy as jnp
import pytest

from evox import State, algorithms, problems, workflows, use_state
from evox.monitors import EvalMonitor, PopMonitor


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
    monitor.set_opt_direction(1)

    pop1 = jnp.arange(15).reshape((3, 5))
    fitness1 = jnp.arange(3)

    state = monitor.init()

    state = use_state(monitor.post_ask)(state, state, pop1)
    state = use_state(monitor.post_eval)(state, state, fitness1)
    state = state.execute_callbacks()
    assert monitor.get_best_fitness(state)[0] == 0
    assert (monitor.get_topk_fitness(state)[0] == fitness1[:topk]).all()
    assert (monitor.get_best_solution(state)[0] == pop1[0]).all()
    assert (monitor.get_topk_solutions(state)[0] == pop1[:topk]).all()

    pop2 = -jnp.arange(15).reshape((3, 5))
    fitness2 = -jnp.arange(3)
    state = use_state(monitor.post_ask)(state, state, pop2)
    state = use_state(monitor.post_eval)(state, state, fitness2)
    state = state.execute_callbacks()
    assert monitor.get_best_fitness(state)[0] == -2
    assert (monitor.get_topk_fitness(state)[0] == fitness2[-topk:][::-1]).all()
    assert (monitor.get_best_solution(state)[0] == pop2[-1]).all()
    assert (monitor.get_topk_solutions(state)[0] == pop2[-topk:][::-1]).all()

    if full_fit_history:
        monitor.plot(state)


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
    monitor.set_opt_direction(1)

    pop1 = jnp.arange(15).reshape((3, 5))
    fitness1 = jnp.arange(6).reshape(3, 2)

    state = monitor.init()

    state = use_state(monitor.post_ask)(state, state, pop1)
    state = use_state(monitor.post_eval)(state, state, fitness1)
    state = state.execute_callbacks()
    assert (monitor.get_latest_fitness(state)[0] == fitness1).all()
    assert (monitor.get_latest_solution(state)[0] == pop1).all()

    pop2 = -jnp.arange(15).reshape((3, 5))
    fitness2 = -jnp.arange(6).reshape(3, 2)
    state = use_state(monitor.post_ask)(state, state, pop2)
    state = use_state(monitor.post_eval)(state, state, fitness2)
    state = state.execute_callbacks()
    assert (monitor.get_latest_fitness(state)[0] == fitness2).all()
    assert (monitor.get_latest_solution(state)[0] == pop2).all()

    if full_fit_history:
        monitor.plot(state)


@pytest.mark.parametrize(
    "full_fit_history,full_pop_history",
    [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ],
)
def test_pop_monitor(full_fit_history, full_pop_history):
    monitor = PopMonitor(
        full_pop_history=full_pop_history, full_fit_history=full_fit_history
    )
    algorithm = algorithms.CSO(lb=jnp.zeros((5,)), ub=jnp.ones((5,)), pop_size=4)
    problem = problems.numerical.Sphere()
    workflow = workflows.StdWorkflow(algorithm, problem, monitors=[monitor])
    key = jax.random.PRNGKey(0)
    state = workflow.init(key)
    state = workflow.step(state)
    assert (
        use_state(monitor.get_latest_fitness)(state)[0]
        == state.get_child_state("algorithm").fitness
    ).all()
    assert (
        use_state(monitor.get_latest_population)(state)[0]
        == state.get_child_state("algorithm").population
    ).all()

    if full_fit_history:
        monitor.plot(state)
