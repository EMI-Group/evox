import chex
import jax
import jax.numpy as jnp
import pytest
from evox import problems


def test_ackley():
    ackley = problems.numerical.Ackley()
    key = jax.random.PRNGKey(12345)
    keys = jax.random.split(key, 16)
    state = ackley.init(keys)
    X = jnp.zeros((16, 2))
    F, state = ackley.evaluate(state, X)
    chex.assert_trees_all_close(F, jnp.zeros((16,)), atol=1e-6)


def test_griewank():
    griewank = problems.numerical.Griewank()
    key = jax.random.PRNGKey(12345)
    keys = jax.random.split(key, 2)
    state = griewank.init(keys)
    X = jnp.zeros((16, 2))
    F, state = griewank.evaluate(state, X)
    chex.assert_trees_all_close(F, jnp.zeros((16,)), atol=1e-6)


def test_rastrigin():
    rastrigin = problems.numerical.Rastrigin()
    key = jax.random.PRNGKey(12345)
    keys = jax.random.split(key, 16)
    state = rastrigin.init(keys)
    X = jnp.zeros((16, 2))
    F, state = rastrigin.evaluate(state, X)
    chex.assert_trees_all_close(F, jnp.zeros((16,)), atol=1e-6)


def test_rosenbrock():
    rosenbrock = problems.numerical.Rosenbrock()
    key = jax.random.PRNGKey(12345)
    keys = jax.random.split(key, 16)
    state = rosenbrock.init(keys)
    X = jnp.ones((16, 2))
    F, state = rosenbrock.evaluate(state, X)
    chex.assert_trees_all_close(F, jnp.zeros((16, )), atol=1e-6)


def test_dtlz1():
    dtlz1 = problems.numerical.DTLZ1(d=None, m=4)
    key = jax.random.PRNGKey(12345)
    keys = jax.random.split(key, 16)
    state = dtlz1.init(keys)
    X = jnp.ones((16, 7))*0.5
    F, state = dtlz1.evaluate(state, X)
    pf, state = dtlz1.pf(state)
    print(pf.shape)
    chex.assert_trees_all_close(
        jnp.sum(F, axis=1), 0.5*jnp.ones((16, )), atol=1e-6)
