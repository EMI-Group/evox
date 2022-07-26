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
