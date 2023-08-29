import jax
import jax.numpy as jnp
import chex
from evox import operators


def test_crowding_distance1():
    x = jnp.array([[0.5, 4], [1, 2.5], [2, 2], [3, 1], [4, 0.8]])
    distance = operators.crowding_distance(x)
    chex.assert_trees_all_close(
        distance,
        jnp.array(
            [
                jnp.inf,
                1.5 / 3.5 + 2 / 3.2,
                2 / 3.5 + 1.5 / 3.2,
                2 / 3.5 + 1.2 / 3.2,
                jnp.inf,
            ]
        ),
    )


def test_crowding_distance2():
    key = jax.random.PRNGKey(314)
    x = jax.random.normal(key, (128, 8))
    rank = operators.non_dominated_sort(x)
    pareto_front = x[rank == 0]
    distance = operators.crowding_distance(pareto_front)
    ground_truth = jnp.array(
        [
            jnp.inf,
            0.11163732,
            0.11454596,
            0.18278188,
            0.24680759,
            jnp.inf,
            0.18060488,
            0.11818929,
            0.15220208,
            0.10824727,
            0.15632743,
            0.06637436,
            0.19527058,
            0.14661205,
            0.15482774,
            0.16319004,
            0.12434653,
            0.13597573,
            0.10219027,
            0.22283527,
            0.21533534,
            0.13833922,
            0.19014072,
            0.20249744,
            jnp.inf,
            0.15004732,
            0.14324155,
            0.20420349,
            0.15499896,
            0.13776696,
            0.22745606,
            jnp.inf,
            0.13295922,
            jnp.inf,
            0.3060898,
            0.12899876,
            0.09699774,
            0.11350346,
            0.29929417,
            jnp.inf,
            0.10250087,
            0.10542361,
            0.19804388,
            jnp.inf,
            0.15468453,
            0.33760267,
            jnp.inf,
            jnp.inf,
            0.34731367,
            0.25738385,
            0.1643093,
            0.1823765,
            0.28458792,
            0.16398923,
            0.27784315,
            0.12338021,
            0.18310618,
            0.1535273,
            0.16749862,
            0.3056032,
            jnp.inf,
            0.1582007,
            0.14821602,
            0.17938283,
            0.32685894,
            0.24114175,
            0.3324114,
            jnp.inf,
            0.12546016,
            0.09681451,
            0.12812261,
            0.14821297,
            jnp.inf,
            0.25850004,
            0.09021468,
            0.14668211,
            0.11086786,
            0.20055725,
            jnp.inf,
            0.20366581,
            0.18799351,
            0.25190017,
            0.12983114,
            0.09826535,
            0.25846872,
            0.11703123,
            jnp.inf,
            0.15235177,
            jnp.inf,
            0.29857454,
        ]
    )
    chex.assert_trees_all_close(distance, ground_truth, rtol=1e-5)


def test_masked_crowding_distance1():
    x = jnp.array([[-1, -1], [0.5, 4], [1, 2.5], [2, 2], [-2, -2], [3, 1], [4, 0.8], [-3, -3], [-3, -4]])
    mask = jnp.array([False, True,     True,     True,   False,    True,   True,     False,    False])
    distance = operators.crowding_distance(x, mask)
    chex.assert_trees_all_close(
        distance,
        jnp.array(
            [
                -jnp.inf,
                jnp.inf,
                1.5 / 3.5 + 2 / 3.2,
                2 / 3.5 + 1.5 / 3.2,
                -jnp.inf,
                2 / 3.5 + 1.2 / 3.2,
                jnp.inf,
                -jnp.inf,
                -jnp.inf,
            ]
        ),
    )