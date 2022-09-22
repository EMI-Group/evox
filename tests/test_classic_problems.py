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
    print(keys)
    state = griewank.init(keys)
    print(state)
    X = jnp.zeros((2, 2))
    state, F = griewank.evaluate(state, X)
    print(F)
    chex.assert_tree_all_close(F, jnp.zeros((2,)), atol=1e-6)

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