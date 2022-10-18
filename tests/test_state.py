import chex
import evoxlib as exl
import jax
import pytest
from evoxlib import Stateful, State
from jax.tree_util import tree_map


class Leaf(Stateful):
    def setup(self, key):
        return State(c=42)

    def run(self, state):
        c = state.c
        return state.update(c=c*2), [2, 7, 1]

    def check(self, state):
        assert state.c == 84
        return state


class Middle(Stateful):
    def __init__(self):
        super().__init__()
        self.leaf = Leaf()

    def setup(self, key):
        return State(d=[3, 1, 4, 1, 5, 9, 2, 6])

    def run(self, state):
        state, e = self.leaf.run(state)
        return state.update(d=3.1415926), e + [8, 2, 8]

    def check(self, state):
        assert state.d == 3.1415926
        return state


class Root(Stateful):
    def __init__(self):
        super().__init__()
        self.a = 123
        self.b = 456
        self.leaf = Leaf()
        self.middle = Middle()

    def setup(self, key):
        return State(attr_a=self.a, attr_b=self.b)

    def run(self, state):
        attr_a = state["attr_a"] + 2
        attr_b = state["attr_b"] - 3
        state, e1 = self.leaf.run(state)
        state, e2 = self.middle.run(state)
        return state | {"attr_a": attr_a, "attr_b": attr_b}, e1 + e2

    def check(self, state):
        assert state.attr_a == 125
        assert state.attr_b == 453

        state = self.middle.check(state)
        state = self.leaf.check(state)
        return state


def test_basic():
    root_module = Root()
    middle_module = Middle()
    leaf_module = Leaf()
    root_state = root_module.init(key=jax.random.PRNGKey(123))
    middle_state = middle_module.init(key=jax.random.PRNGKey(456))
    leaf_state = leaf_module.init(key=jax.random.PRNGKey(789))
    assert root_state.get_child_state('_submodule_leaf') == leaf_state
    assert root_state.get_child_state('_submodule_middle') == middle_state
    assert middle_state.get_child_state('_submodule_leaf') == leaf_state

    root_state, magic = root_module.run(root_state)
    assert magic == [2, 7, 1, 2, 7, 1, 8, 2, 8]
    root_state = root_module.check(root_state)


def test_repl_and_str():
    module = Root()
    state = module.init(key=jax.random.PRNGKey(456))
    assert repr(state) == "State ({'attr_a': 123, 'attr_b': 456}, ['_submodule_leaf', '_submodule_middle'])"
    assert str(state) == ("State (\n"
                          " {'attr_a': 123, 'attr_b': 456},\n"
                          " ['_submodule_leaf', '_submodule_middle']\n"
                          ")"
                          )

def test_jax_pytree():
    module = Root()
    state = module.init(key=jax.random.PRNGKey(0))
    assert state == tree_map(lambda x: x, state)
