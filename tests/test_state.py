import chex
import evoxlib as exl
import jax
import jax.numpy as jnp
import pytest


class Leaf(exl.Module):
    def setup(self, key):
        return {"c": jnp.arange(10)}

    def run(self, state):
        c = state["c"]
        return {"c": c * 2}


class Root(exl.Module):
    def __init__(self):
        self.a = 123
        self.b = 456
        self.leaf = Leaf()

    def setup(self, key):
        return {"attr_a": self.a, "attr_b": self.b}

    def run(self, state):
        attr_a = state["attr_a"] + 2
        attr_b = state["attr_b"] - 2
        state = self.leaf.run(state)
        return state | {"attr_a": attr_a, "attr_b": attr_b}, 789

# disable it for now
def __test_basic():
    key = jax.random.PRNGKey(123)
    test_module = Root()
    state = test_module.init(key=key)
    chex.assert_trees_all_close(
        state, {"attr_a": 123, "attr_b": 456, "_submodule_leaf": {"c": jnp.arange(10)}}
    )
    state, magic = test_module.run(state)
    chex.assert_trees_all_close(
        state,
        state,
        {"attr_a": 125, "attr_b": 454, "_submodule_leaf": {"c": jnp.arange(10) * 2}},
    )
    assert magic == 789
