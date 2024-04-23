import jax
import jax.numpy as jnp
import pytest
from evox import workflows, algorithms, problems
from evox.monitors import StdSOMonitor, StdMOMonitor, EvalMonitor


@pytest.mark.parametrize("topk", [1, 2, 4])
def test_std_so_monitor(topk):
    monitor = StdSOMonitor(record_topk=topk, record_fit_history=True)

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

    monitor.post_eval(None, pop1, None, fitness1)
    assert monitor.get_best_fitness() == 0
    assert (monitor.get_topk_fitness() == fitness1[:topk]).all()
    assert (monitor.get_best_solution() == pop1[0]).all()
    assert (monitor.get_topk_solutions() == pop1[:topk]).all()

    pop2 = -jnp.arange(15).reshape((3, 5))
    fitness2 = -jnp.arange(3)
    monitor.post_eval(None, pop2, None, fitness2)
    assert monitor.get_best_fitness() == -2
    assert (monitor.get_topk_fitness() == fitness2[-topk:][::-1]).all()
    assert (monitor.get_best_solution() == pop2[-1]).all()
    assert (monitor.get_topk_solutions() == pop2[-topk:][::-1]).all()
