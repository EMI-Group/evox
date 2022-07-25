import evoxlib as exl
import jax.numpy as jnp
import chex
import pytest

class Leaf(exl.module.Module):
    def setup(self):
        return {
            'c': jnp.arange(10)
        }

    @exl.module.lift
    def run(self, state):
        c = state['c']
        return {
            'c': c * 2
        }

class Root(exl.module.Module):
    def __init__(self):
        self.a = 123
        self.b = 456
        self.leaf = Leaf()

    def setup(self):
        return {
            'attr_a': self.a,
            'attr_b': self.b
        }

    @exl.module.lift
    def run(self, state):
        attr_a = state['attr_a'] + 2
        attr_b = state['attr_b'] - 2
        state = self.leaf.run(state)
        return state | {'attr_a': attr_a, 'attr_b': attr_b}

def test_basic():
    test_module = Root()
    state = test_module.init()
    chex.assert_trees_all_close(state, {
        '_sub_leaf': {
            'c': jnp.arange(10)
        },
        'attr_a': 123,
        'attr_b': 456
    })
    state = test_module.run(state)
    chex.assert_trees_all_close(state, {
        '_sub_leaf': {
            'c': jnp.arange(10) * 2
        },
        'attr_a': 125,
        'attr_b': 454
    })