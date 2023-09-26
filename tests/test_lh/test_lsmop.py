import jax.numpy as jnp
import jax.random
from jax import random
import sys
from matplotlib import path
import numpy as np
from matplotlib.path import Path
from evox.problems.numerical.lsmop import *
import pytest


def test_lsmop1():
    key = random.PRNGKey(1)
    data = jnp.load("data.npz")['X']
    data = jnp.array(data)
    [n, d] = data.shape
    m = 6
    keys = jax.random.split(key, 16)
    prob = LSMOP1(d=d, m=m)
    if d != prob.d:
        if d < prob.d:
            pad_width = [(0, 0), (0, int(prob.d - d))]
            data = jnp.pad(data, pad_width, mode='wrap')
        else:
            data = data[:, :prob.d]
    state = prob.init(keys)
    r1, new_state1 = prob.evaluate(state, data)
    r2, new_state2 = prob.pf(state)
    assert r1.shape[0] == 105
    assert r2.shape[1] == m

def test_lsmop2():
    key = random.PRNGKey(1)
    data = jnp.load("data.npz")['X']
    data = jnp.array(data)
    [n, d] = data.shape
    m = 6
    keys = jax.random.split(key, 16)
    prob = LSMOP2(d=d, m=m)
    if d != prob.d:
        if d < prob.d:
            pad_width = [(0, 0), (0, int(prob.d - d))]
            data = jnp.pad(data, pad_width, mode='wrap')
        else:
            data = data[:, :prob.d]
    state = prob.init(keys)
    r1, new_state1 = prob.evaluate(state, data)
    r2, new_state2 = prob.pf(state)
    assert r1.shape[0] == 105
    assert r2.shape[1] == m

def test_lsmop3():
    key = random.PRNGKey(1)
    data = jnp.load("data.npz")['X']
    data = jnp.array(data)
    [n, d] = data.shape
    m = 6
    keys = jax.random.split(key, 16)
    prob = LSMOP4(d=d, m=m)
    if d != prob.d:
        if d < prob.d:
            pad_width = [(0, 0), (0, int(prob.d - d))]
            data = jnp.pad(data, pad_width, mode='wrap')
        else:
            data = data[:, :prob.d]
    state = prob.init(keys)
    r1, new_state1 = prob.evaluate(state, data)
    r2, new_state2 = prob.pf(state)
    assert r1.shape[0] == 105
    assert r2.shape[1] == m

def test_lsmop4():
    key = random.PRNGKey(1)
    data = jnp.load("data.npz")['X']
    data = jnp.array(data)
    [n, d] = data.shape
    m = 6
    keys = jax.random.split(key, 16)
    prob = LSMOP4(d=d, m=m)
    if d != prob.d:
        if d < prob.d:
            pad_width = [(0, 0), (0, int(prob.d - d))]
            data = jnp.pad(data, pad_width, mode='wrap')
        else:
            data = data[:, :prob.d]
    state = prob.init(keys)
    r1, new_state1 = prob.evaluate(state, data)
    r2, new_state2 = prob.pf(state)
    assert r1.shape[0] == 105
    assert r2.shape[1] == m

def test_lsmop5():
    key = random.PRNGKey(1)
    data = jnp.load("data.npz")['X']
    data = jnp.array(data)
    [n, d] = data.shape
    m = 6
    keys = jax.random.split(key, 16)
    prob = LSMOP5(d=d, m=m)
    if d != prob.d:
        if d < prob.d:
            pad_width = [(0, 0), (0, int(prob.d - d))]
            data = jnp.pad(data, pad_width, mode='wrap')
        else:
            data = data[:, :prob.d]
    state = prob.init(keys)
    r1, new_state1 = prob.evaluate(state, data)
    r2, new_state2 = prob.pf(state)
    assert r1.shape[0] == 105
    assert r2.shape[1] == m

def test_lsmop6():
    key = random.PRNGKey(1)
    data = jnp.load("data.npz")['X']
    data = jnp.array(data)
    [n, d] = data.shape
    m = 6
    keys = jax.random.split(key, 16)
    prob = LSMOP6(d=d, m=m)
    if d != prob.d:
        if d < prob.d:
            pad_width = [(0, 0), (0, int(prob.d - d))]
            data = jnp.pad(data, pad_width, mode='wrap')
        else:
            data = data[:, :prob.d]
    state = prob.init(keys)
    r1, new_state1 = prob.evaluate(state, data)
    r2, new_state2 = prob.pf(state)
    assert r1.shape[0] == 105
    assert r2.shape[1] == m

def test_lsmop7():
    key = random.PRNGKey(1)
    data = jnp.load("data.npz")['X']
    data = jnp.array(data)
    [n, d] = data.shape
    m = 6
    keys = jax.random.split(key, 16)
    prob = LSMOP7(d=d, m=m)
    if d != prob.d:
        if d < prob.d:
            pad_width = [(0, 0), (0, int(prob.d - d))]
            data = jnp.pad(data, pad_width, mode='wrap')
        else:
            data = data[:, :prob.d]
    state = prob.init(keys)
    r1, new_state1 = prob.evaluate(state, data)
    r2, new_state2 = prob.pf(state)
    assert r1.shape[0] == 105
    assert r2.shape[1] == m

def test_lsmop8():
    key = random.PRNGKey(1)
    data = jnp.load("data.npz")['X']
    data = jnp.array(data)
    [n, d] = data.shape
    m = 6
    keys = jax.random.split(key, 16)
    prob = LSMOP8(d=d, m=m)
    if d != prob.d:
        if d < prob.d:
            pad_width = [(0, 0), (0, int(prob.d - d))]
            data = jnp.pad(data, pad_width, mode='wrap')
        else:
            data = data[:, :prob.d]
    state = prob.init(keys)
    r1, new_state1 = prob.evaluate(state, data)
    r2, new_state2 = prob.pf(state)
    assert r1.shape[0] == 105
    assert r2.shape[1] == m

def test_lsmop9():
    key = random.PRNGKey(1)
    data = jnp.load("data.npz")['X']
    data = jnp.array(data)
    [n, d] = data.shape
    m = 6
    keys = jax.random.split(key, 16)
    prob = LSMOP9(d=d, m=m)
    if d != prob.d:
        if d < prob.d:
            pad_width = [(0, 0), (0, int(prob.d - d))]
            data = jnp.pad(data, pad_width, mode='wrap')
        else:
            data = data[:, :prob.d]
    state = prob.init(keys)
    r1, new_state1 = prob.evaluate(state, data)
    r2, new_state2 = prob.pf(state)
    assert r1.shape[0] == 105
    assert r2.shape[1] == m