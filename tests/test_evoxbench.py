import jax
import jax.numpy as jnp
import pytest
from evox import problems


@pytest.mark.skip(reason="requires downloading the evoxbench database")
def test_c10mop():
    c10mop = problems.evoxbench.C10MOP(1)
    key = jax.random.PRNGKey(12345)
    state = c10mop.init(key)
    X = jnp.array([c10mop.lb, c10mop.ub])
    fitness, state = c10mop.evaluate(state, X)
    assert fitness.shape == (2, c10mop.n_objs)


@pytest.mark.skip(reason="requires downloading the evoxbench database")
def test_citysegmop():
    cityseg = problems.evoxbench.CitySegMOP(1)
    key = jax.random.PRNGKey(12345)
    state = cityseg.init(key)
    X = jnp.array([cityseg.lb, cityseg.ub])
    fitness, state = cityseg.evaluate(state, X)
    assert fitness.shape == (2, cityseg.n_objs)


@pytest.mark.skip(reason="requires downloading the evoxbench database")
def test_in10kmop():
    in10kmop = problems.evoxbench.IN1kMOP(1)
    key = jax.random.PRNGKey(12345)
    state = in10kmop.init(key)
    X = jnp.array([in10kmop.lb, in10kmop.ub])
    fitness, state = in10kmop.evaluate(state, X)
    assert fitness.shape == (2, in10kmop.n_objs)
