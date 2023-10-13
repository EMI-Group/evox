from jax import random
from evox.problems.numerical.maf import *
import pytest
import time

"""
x is input data, which default in [0, 1]
specifically: in MaF8, MaF9, x is in [-10000, 10000]; in MaF10, MaF11, MaF12, x is in [0, 2]
"""
key = random.PRNGKey(1)
data = random.uniform(key, (3, 12))
n, d = data.shape
m = 3

def test_inside():
    assert inside(8.5, 1.0, 0.0) == False
    assert inside(8.5, 0.0, 1.0) == False
    assert inside(0.5, 0.0, 1.0) == True
    assert inside(0.5, 1.0, 0.0) == True
    assert inside(1.0, 1.0, 0.0) == False
    assert inside(0.0, 1.0, 0.0) == True
    assert inside(1.0, 0.0, 1.0) == False
    assert inside(0.0, 0.0, 1.0) == True

def test_ray_intersect_segment():
    point = jnp.array([0.0, 0.0])
    assert ray_intersect_segment(point, jnp.array([1.0, 1.0]), jnp.array([1.0, 2.0])) == False
    assert ray_intersect_segment(point, jnp.array([1.0, 1.0]), jnp.array([-1.0, -1.0])) == True
    assert ray_intersect_segment(point, jnp.array([1.0, 1.0]), jnp.array([1.0, -1.0])) == True
    assert ray_intersect_segment(point, jnp.array([1.0, 0.0]), jnp.array([1.0, -1.0])) == False
    assert ray_intersect_segment(point, jnp.array([1.0, 0.0]), jnp.array([1.0, 1.0])) == True
    assert ray_intersect_segment(point, jnp.array([1.0, 1.0]), jnp.array([1.0, 0.0])) == True

def test_point_in_polygon():
    polygon = jnp.array([
        [0, 1.0],
        [-0.5, -1],
        [0.5, -1],
    ])
    point = jnp.array([0, 0])
    assert point_in_polygon(polygon, point) == True
    point = jnp.array([1, -1])
    assert point_in_polygon(polygon, point) == False
    point = jnp.array([0, 1.0])
    assert point_in_polygon(polygon, point) == True
    point = jnp.array([-1, 1.0])
    assert point_in_polygon(polygon, point) == False

def test_maf1():
    prob = MaF1(d=d, m=m)
    state = prob.init(key)
    r1, new_state1 = prob.evaluate(state, data)
    r2, new_state2 = prob.pf(state)
    assert r1.shape == (3, 3)
    assert abs(float(r1[1, 1]) - 1.8404) < 0.0001
    assert r2.shape[1] == 3
    assert float(r2[1, 1]) - 0.9867 < 0.0001


def test_maf2():
    prob = MaF2(d=d, m=m)
    state = prob.init(key)
    r1, new_state1 = prob.evaluate(state, data)
    r2, new_state2 = prob.pf(state)
    assert r1.shape == (3, 3)
    assert abs(float(r1[1, 1]) - 0.6237) < 0.0001
    assert r2.shape[1] == 3
    assert float(r2[1, 1]) - 0.3458 < 0.0001


def test_maf3():
    prob = MaF3(d=d, m=m)
    state = prob.init(key)
    r1, new_state1 = prob.evaluate(state, data)
    r2, new_state2 = prob.pf(state)
    assert r1.shape == (3, 3)
    assert abs(float(r1[1, 1]) - 2.1354973e11) / 2.1354973e11 < 0.0001
    assert r2.shape[1] == 3
    assert float(r2[1, 1]) - 1.8255e-04 < 0.0001


def test_maf4():
    prob = MaF4(d=d, m=m)
    state = prob.init(key)
    r1, new_state1 = prob.evaluate(state, data)
    r2, new_state2 = prob.pf(state)
    assert r1.shape == (3, 3)
    assert abs(float(r1[1, 1]) - 1.9944e03) < 0.01
    assert r2.shape[1] == 3
    assert abs(float(r2[1, 1]) - 3.9460) < 0.0001


def test_maf5():
    prob = MaF5(d=d, m=m)
    state = prob.init(key)
    r1, new_state1 = prob.evaluate(state, data)
    r2, new_state2 = prob.pf(state)
    assert r1.shape == (3, 3)
    assert abs(float(r1[1, 1]) - 2.1819e-40) < 0.0001
    assert r2.shape[1] == 3
    assert abs(float(r2[1, 1]) - 0.0540) < 0.0001


def test_maf6():
    prob = MaF6(d=d, m=m)
    state = prob.init(key)
    r1, new_state1 = prob.evaluate(state, data)
    r2, new_state2 = prob.pf(state)
    assert r1.shape == (3, 3)
    assert abs(float(r1[1, 1]) - 55.8732) < 0.0001
    assert r2.shape[1] == 3
    assert abs(float(r2[1, 1]) - 2.3586e-04) < 0.0001


def test_maf7():
    prob = MaF7(d=d, m=m)
    state = prob.init(key)
    r1, new_state1 = prob.evaluate(state, data)
    r2, new_state2 = prob.pf(state)
    assert r1.shape == (3, 3)
    assert abs(float(r1[1, 1]) - 0.3915) < 0.0001
    assert r2.shape[1] == 3
    assert abs(float(r2[1, 1]) - 0) < 0.0001


def test_maf8():
    prob = MaF8(d=d, m=m)
    state = prob.init(key)
    r1, new_state1 = prob.evaluate(state, data)
    r2, new_state2 = prob.pf(state)
    assert r1.shape == (3, 3)
    assert abs(float(r1[1, 1]) - 1.2490) < 0.0001
    assert r2.shape[1] == 3
    assert abs(float(r2[1, 1]) - 0.0545) < 0.0001


def test_maf9():
    prob = MaF9(d=d, m=m)
    state = prob.init(key)
    r1, new_state1 = prob.evaluate(state, data)
    r2, new_state2 = prob.pf(state)
    assert r1.shape == (3, 3)
    assert abs(float(r1[1, 1]) - 0.3118) / abs(float(r1[1, 1])) < 0.0001
    assert r2.shape[1] == 3
    assert abs(float(r2[1, 1]) - 0.0351) < 0.0001


def test_maf10():
    prob = MaF10(d=d, m=m)
    state = prob.init(key)
    r1, new_state1 = prob.evaluate(state, data)
    r2, new_state2 = prob.pf(state)
    assert r1.shape == (3, 3)
    assert abs(float(r1[1, 1]) - 1.0060) < 0.0001
    assert r2.shape[1] == 3
    assert abs(float(r2[1, 1]) - 0.0483) < 0.0001


def test_maf11():
    prob = MaF11(d=d, m=m)
    state = prob.init(key)
    r1, new_state1 = prob.evaluate(state, data)
    r2, new_state2 = prob.pf(state)
    assert r1.shape == (3, 3)
    assert abs(float(r1[1, 1]) - 0.6342) < 0.0001
    assert r2.shape[1] == 3
    assert abs(float(r2[1, 1]) - 0.1021) < 0.0001


def test_maf12():
    prob = MaF12(d=d, m=m)
    state = prob.init(key)
    r1, new_state1 = prob.evaluate(state, data)
    r2, new_state2 = prob.pf(state)
    assert r1.shape == (3, 3)
    assert abs(float(r1[1, 1]) - 3.4703) < 0.0001
    assert r2.shape[1] == 3
    assert abs(float(r2[1, 1]) - 0.0540) < 0.0001


def test_maf13():
    prob = MaF13(d=d, m=m)
    state = prob.init(key)
    r1, new_state1 = prob.evaluate(state, data)
    r2, new_state2 = prob.pf(state)
    assert r1.shape == (3, 3)
    assert abs(float(r1[1, 1]) - 0.8246) < 0.0001
    assert r2.shape[1] == 3
    assert abs(float(r2[1, 1]) - 0.0135) < 0.0001


def test_maf14():
    prob = MaF14(d=d, m=m)
    state = prob.init(key)
    r1, new_state1 = prob.evaluate(state, data)
    r2, new_state2 = prob.pf(state)
    assert r1.shape == (3, 3)
    assert abs(float(r1[1, 1]) - 0.1718) < 0.0001
    assert r2.shape[1] == 3
    assert abs(float(r2[1, 1]) - 0.0133) < 0.0001


def test_maf15():
    prob = MaF15(d=d, m=m)
    state = prob.init(key)
    r1, new_state1 = prob.evaluate(state, data)
    r2, new_state2 = prob.pf(state)
    assert r1.shape == (3, 3)
    assert abs(float(r1[1, 1]) - 0.8123) < 0.0001
    assert r2.shape[1] == 3
    assert abs(float(r2[1, 1]) - 0.9865) < 0.0001
