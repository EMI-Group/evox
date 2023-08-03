import evox
from evox.operators import non_dominated_sort
import jax
import jax.numpy as jnp
import chex


def test_non_dominated_sort():
    x = jnp.array(
        [
            [0.5, 4],
            [1, 2],
            [2, 1],
            [3, 1],
            [4, 1],
            [5, 0.5],
            [1, 4],
            [2, 3],
            [3, 2],
            [3, 3],
            [4, 4],
        ]
    )

    rank = non_dominated_sort(x)
    chex.assert_trees_all_close(
        rank, jnp.array([0, 0, 0, 1, 2, 0, 1, 1, 2, 3, 4], dtype=jnp.int32)
    )
