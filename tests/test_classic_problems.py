import chex
import evoxlib as exl
import jax
import jax.numpy as jnp
import pytest

def test_ackley():
    ackley = exl.problems.classic.Ackley()
    key = jax.random.PRNGKey(12345)
    state = ackley.init()
    X = jnp.zeros((16, ))
    state, F = ackley.evaluate(state, X)
    chex.assert_tree_all_close(F, jnp.zeros((1, )), atol=1e-6)