import chex
import evoxlib as exl
import jax
import jax.numpy as jnp
import pytest


def test_ackley():
    ackley = exl.problems.classic.Ackley()
    key = jax.random.PRNGKey(12345)
    keys = jax.random.split(key, 16)
    state = ackley.init(keys)
    X = jnp.zeros((16, 2))
    state, F = ackley.evaluate(state, X)
    chex.assert_tree_all_close(F, jnp.zeros((16,)), atol=1e-6)


def test_griewank():
    griewank = exl.problems.classic.Griewank()
    key = jax.random.PRNGKey(12345)
    keys = jax.random.split(key, 2)
    state = griewank.init(keys)
    X = jnp.zeros((16, 2))
    state, F = griewank.evaluate(state, X)
    chex.assert_tree_all_close(F, jnp.zeros((16,)), atol=1e-6)


def test_rastrigin():
    rastrigin = exl.problems.classic.Rastrigin()
    key = jax.random.PRNGKey(12345)
    keys = jax.random.split(key, 16)
    state = rastrigin.init(keys)
    X = jnp.zeros((16, 2))
    state, F = rastrigin.evaluate(state, X)
    chex.assert_tree_all_close(F, jnp.zeros((16,)), atol=1e-6)


def test_rosenbrock():
    rosenbrock = exl.problems.classic.Rosenbrock()
    key = jax.random.PRNGKey(12345)
    keys = jax.random.split(key, 16)
    state = rosenbrock.init(keys)
    X = jnp.ones((16, 2))
    state, F = rosenbrock.evaluate(state, X)
    chex.assert_tree_all_close(F, jnp.zeros((16, )), atol=1e-6)


def test_dtlz1():
    dtlz1 = exl.problems.classic.DTLZ1(d=None, m=4)
    key = jax.random.PRNGKey(12345)
    keys = jax.random.split(key, 16)
    state = dtlz1.init(keys)
    X = jnp.ones((16, 7))*0.5
    state, F = dtlz1.evaluate(state, X)
    state, pf = dtlz1.pf(state)
    print(pf.shape)
    chex.assert_tree_all_close(
        jnp.sum(F, axis=1), 0.5*jnp.ones((16, )), atol=1e-6)
