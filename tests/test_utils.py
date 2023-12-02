import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
import chex
import pytest
from evox import utils

def assert_invertible(tree):
    tree_to_vector = utils.TreeAndVector(tree)
    vector = tree_to_vector.to_vector(tree)
    tree2 = tree_to_vector.to_tree(vector)
    chex.assert_trees_all_close(tree, tree2)

    tree_batched = tree_map(lambda x, y: jnp.stack([x, y], axis=0), tree, tree)
    vector_batched = tree_to_vector.batched_to_vector(tree_batched)
    tree_batched2 = tree_to_vector.batched_to_tree(vector_batched)
    chex.assert_trees_all_close(tree_batched, tree_batched2)

def test_tree_and_vector():
    tree = {
        'layer1': jnp.arange(10).reshape(2, 5),
        'layer2': {
            'layer2_1': jnp.arange(10).reshape(2, 5),
            'layer2_2': jnp.arange(8).reshape(2, 2, 2),
            'layer2_3': jnp.arange(9).reshape(1, 1, 1, 9),
        },
        'layer3': [
            jnp.ones((7, 7)),
            jnp.zeros((8, 8))
        ]
    }
    assert_invertible(tree)