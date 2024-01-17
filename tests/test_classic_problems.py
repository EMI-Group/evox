import chex
import jax
import jax.numpy as jnp
import pytest
from evox import problems


def test_ackley():
    ackley = problems.numerical.Ackley()
    key = jax.random.PRNGKey(12345)
    state = ackley.init(key)
    X = jnp.zeros((16, 2))
    F, state = ackley.evaluate(state, X)
    chex.assert_trees_all_close(F, jnp.zeros((16,)), atol=1e-6)


def test_griewank():
    griewank = problems.numerical.Griewank()
    key = jax.random.PRNGKey(12345)
    state = griewank.init(key)
    X = jnp.zeros((16, 2))
    F, state = griewank.evaluate(state, X)
    chex.assert_trees_all_close(F, jnp.zeros((16,)), atol=1e-6)


def test_rastrigin():
    rastrigin = problems.numerical.Rastrigin()
    key = jax.random.PRNGKey(12345)
    state = rastrigin.init(key)
    X = jnp.zeros((16, 2))
    F, state = rastrigin.evaluate(state, X)
    chex.assert_trees_all_close(F, jnp.zeros((16,)), atol=1e-6)


def test_rosenbrock():
    rosenbrock = problems.numerical.Rosenbrock()
    key = jax.random.PRNGKey(12345)
    state = rosenbrock.init(key)
    X = jnp.ones((16, 2))
    F, state = rosenbrock.evaluate(state, X)
    chex.assert_trees_all_close(F, jnp.zeros((16,)), atol=1e-6)


def test_schwefel():
    schwefel = problems.numerical.Schwefel()
    key = jax.random.PRNGKey(12345)
    state = schwefel.init(key)
    X = jnp.full(shape=(2, 2), fill_value=420.9687462275036)
    F, state = schwefel.evaluate(state, X)
    chex.assert_trees_all_close(F, jnp.zeros((2,)), atol=1e-4)


def test_dtlz1():
    dtlz1 = problems.numerical.DTLZ1(d=None, m=4)
    key = jax.random.PRNGKey(12345)
    state = dtlz1.init(key)
    X = jnp.ones((16, 7)) * 0.5
    F, state = dtlz1.evaluate(state, X)
    pf, state = dtlz1.pf(state)
    print(pf.shape)
    chex.assert_trees_all_close(jnp.sum(F, axis=1), 0.5 * jnp.ones((16,)), atol=1e-6)
