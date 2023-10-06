import jax.numpy as jnp
import jax.random
from jax import random
from evox.problems.numerical.lsmop import *
import pytest


"""
the range of x are [0, 1] in first (M - 1) dimensions, and [0,10] in other dimensions. 
"""

key = random.PRNGKey(0)
upper1 = jnp.ones(2)
upper2 = 10 * jnp.ones(298)
upper = jnp.hstack((upper1, upper2))
data = random.uniform(key, (100, 300), minval=0, maxval=upper)
n, d = data.shape
m = 3
keys = jax.random.split(key, 16)


def test_lsmop1():
    global data, keys, d, m
    prob = LSMOP1(d=d, m=m)
    state = prob.init(keys)
    r1, new_state1 = prob.evaluate(state, data)
    r2, new_state2 = prob.pf(state)
    assert r1.shape == (100, 3)
    assert abs(float(r1[1, 1]) - 12.5981) < 0.0001
    assert r2.shape[1] == 3
    assert float(r2[1, 1]) - 0.0133 < 0.0001


def test_lsmop2():
    global data, keys, d, m
    prob = LSMOP2(d=d, m=m)
    state = prob.init(keys)
    r1, new_state1 = prob.evaluate(state, data)
    r2, new_state2 = prob.pf(state)
    assert r1.shape == (100, 3)
    assert abs(float(r1[1, 1]) - 0.6876) < 0.0001
    assert r2.shape[1] == 3
    assert float(r2[1, 1]) - 0.0133 < 0.0001


def test_lsmop3():
    global data, keys, d, m
    prob = LSMOP3(d=d, m=m)
    state = prob.init(keys)
    r1, new_state1 = prob.evaluate(state, data)
    r2, new_state2 = prob.pf(state)
    assert r1.shape == (100, 3)
    assert abs(float(r1[1, 1]) - 5.3782496e04) / 5.3782496e04 < 0.0001
    assert r2.shape[1] == 3
    assert float(r2[1, 1]) - 0.0133 < 0.0001


def test_lsmop4():
    global data, keys, d, m
    prob = LSMOP4(d=d, m=m)
    state = prob.init(keys)
    r1, new_state1 = prob.evaluate(state, data)
    r2, new_state2 = prob.pf(state)
    assert r1.shape == (100, 3)
    assert abs(float(r1[1, 1]) - 0.5550) < 0.0001
    assert r2.shape[1] == 3
    assert float(r2[1, 1]) - 0.0133 < 0.0001


def test_lsmop5():
    global data, keys, d, m
    prob = LSMOP5(d=d, m=m)
    state = prob.init(keys)
    r1, new_state1 = prob.evaluate(state, data)
    r2, new_state2 = prob.pf(state)
    assert r1.shape == (100, 3)
    assert abs(float(r1[1, 1]) - 2.6036) < 0.0001
    assert r2.shape[1] == 3
    assert float(r2[1, 1]) - 0.0135 < 0.0001


def test_lsmop6():
    global data, keys, d, m
    prob = LSMOP6(d=d, m=m)
    state = prob.init(keys)
    r1, new_state1 = prob.evaluate(state, data)
    r2, new_state2 = prob.pf(state)
    assert r1.shape == (100, 3)
    assert abs(float(r1[1, 1]) - 1.7095365e03) < 0.001
    assert r2.shape[1] == 3
    assert float(r2[1, 1]) - 0.0135 < 0.0001


def test_lsmop7():
    global data, keys, d, m
    prob = LSMOP7(d=d, m=m)
    state = prob.init(keys)
    r1, new_state1 = prob.evaluate(state, data)
    r2, new_state2 = prob.pf(state)
    assert r1.shape == (100, 3)
    assert abs(float(r1[1, 1]) - 1.4722900e04) / 1.4722900e04 < 0.0001
    assert r2.shape[1] == 3
    assert float(r2[1, 1]) - 0.0135 < 0.0001


def test_lsmop8():
    global data, keys, d, m
    prob = LSMOP8(d=d, m=m)
    state = prob.init(keys)
    r1, new_state1 = prob.evaluate(state, data)
    r2, new_state2 = prob.pf(state)
    assert r1.shape == (100, 3)
    assert abs(float(r1[1, 1]) - 1.9358) < 0.0001
    assert r2.shape[1] == 3
    assert float(r2[1, 1]) - 0.0135 < 0.0001


def test_lsmop9():
    global data, keys, d, m
    prob = LSMOP9(d=d, m=m)
    state = prob.init(keys)
    r1, new_state1 = prob.evaluate(state, data)
    r2, new_state2 = prob.pf(state)
    assert r1.shape == (100, 3)
    assert abs(float(r1[1, 1]) - 0.0523) < 0.0001
    assert r2.shape[1] == 3
    assert float(r2[1, 1]) - 0 < 0.0001
