import jax
import jax.numpy as jnp
import pytest
from evox import workflows, algorithms, problems
from evox.monitors import StdSOMonitor, StdMOMonitor


def test_std_so_monitor_top1():
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


def test_std_so_monitor_top2():
    monitor = StdSOMonitor(record_topk=2, record_fit_history=True)

    pop1 = jnp.arange(15).reshape((3, 5))
    fitness1 = jnp.arange(3)
    monitor.record_pop(pop1)
    monitor.record_fit(fitness1)
    assert monitor.get_best_fitness() == 0
    assert (monitor.get_topk_fitness() == jnp.array([0, 1])).all()
    assert (monitor.get_best_solution() == pop1[0]).all()
    assert (monitor.get_topk_solutions() == pop1[0:2]).all()

    pop2 = -jnp.arange(15).reshape((3, 5))
    fitness2 = -jnp.arange(3)
    monitor.record_pop(pop2)
    monitor.record_fit(fitness2)
    assert monitor.get_best_fitness() == -2
    assert (monitor.get_topk_fitness() == jnp.array([-2, -1])).all()
    assert (monitor.get_best_solution() == pop2[2]).all()
    assert (monitor.get_topk_solutions() == pop2[2:0:-1]).all()


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
