from evox.metrics import HV, GD, GDPlus, IGD, IGDPlus
import jax.numpy as jnp
from jax import random
from math import isclose


def test_gd_and_igd():
    pf = jnp.array(
        [
            [0, 5],
            [1, 4],
            [2, 3],
            [3, 2],
            [4, 1],
            [5.5, 0],
        ]
    )
    objs = jnp.array(
        [
            [0, 6],
            [5, 8],
            [4.3, 2],
        ]
    )
    assert isclose(GD(pf)(objs), 2.5669618, rel_tol=1e-4)
    assert isclose(GDPlus(pf)(objs), 2.5669618, rel_tol=1e-4)
    assert isclose(IGD(pf)(objs), 1.7367444, rel_tol=1e-4)
    assert isclose(IGDPlus(pf)(objs), 1.6073387, rel_tol=1e-4)

    # by definition
    assert isclose(GD(pf)(objs), IGD(objs)(pf), abs_tol=1e-4)


def test_hv():
    # hv is designed to work with both max/min optimization problem
    # here it's ok to have a reference point that is smaller than our objectives
    ref = jnp.array([-1, -1])
    objs = jnp.array([
        [1, 9],
        [2, 2],
        [3, 1]
    ])

    key = random.PRNGKey(0)
    hv = HV(ref, 100_000, "bounding_cube")
    assert isclose(hv(key, objs), 25, rel_tol=1e-2)
    hv = HV(ref, 100_000, "each_cube")
    assert isclose(hv(key, objs), 25, rel_tol=1e-2)

    ref = jnp.array([10, 9, 8.0])
    objs = jnp.array([
        [0.1, 7, 0.3],
        [5.5, 0, 2.3],
        [-0.1, 1, -1],
        [3, 2, 5.5],
    ])
    hv = HV(ref, 100_000, "bounding_cube")
    assert isclose(hv(key, objs), 753, rel_tol=1e-2)
    hv = HV(ref, 100_000, "each_cube")
    assert isclose(hv(key, objs), 753, rel_tol=1e-2)