import jax.numpy as jnp
import jax.random
from jax import random
import sys
from matplotlib import path
import numpy as np
from matplotlib.path import Path
from evox.problems.numerical.maf import *
import pytest


def test_maf1():
    key = random.PRNGKey(1)
    data = jnp.load("data.npz")['X']
    data = jnp.array(data)
    [n, d] = data.shape
    m = 6
    keys = jax.random.split(key, 16)
    prob = MaF1(d=d, m=m)
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

def test_maf2():
    key = random.PRNGKey(1)
    data = jnp.load("data.npz")['X']
    data = jnp.array(data)
    [n, d] = data.shape
    m = 6
    keys = jax.random.split(key, 16)
    prob = MaF2(d=d, m=m)
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

def test_maf3():
    key = random.PRNGKey(1)
    data = jnp.load("data.npz")['X']
    data = jnp.array(data)
    [n, d] = data.shape
    m = 6
    keys = jax.random.split(key, 16)
    prob = MaF3(d=d, m=m)
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

def test_maf4():
    key = random.PRNGKey(1)
    data = jnp.load("data.npz")['X']
    data = jnp.array(data)
    [n, d] = data.shape
    m = 6
    keys = jax.random.split(key, 16)
    prob = MaF4(d=d, m=m)
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

def test_maf5():
    key = random.PRNGKey(1)
    data = jnp.load("data.npz")['X']
    data = jnp.array(data)
    [n, d] = data.shape
    m = 6
    keys = jax.random.split(key, 16)
    prob = MaF5(d=d, m=m)
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

def test_maf6():
    key = random.PRNGKey(1)
    data = jnp.load("data.npz")['X']
    data = jnp.array(data)
    [n, d] = data.shape
    m = 6
    keys = jax.random.split(key, 16)
    prob = MaF6(d=d, m=m)
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

def test_maf7():
    key = random.PRNGKey(1)
    data = jnp.load("data.npz")['X']
    data = jnp.array(data)
    [n, d] = data.shape
    m = 6
    keys = jax.random.split(key, 16)
    prob = MaF7(d=d, m=m)
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

def test_maf8():
    key = random.PRNGKey(1)
    data = jnp.load("data.npz")['X']
    data = jnp.array(data)
    [n, d] = data.shape
    m = 6
    keys = jax.random.split(key, 16)
    prob = MaF8(d=d, m=m)
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

def test_maf9():
    key = random.PRNGKey(1)
    data = jnp.load("data.npz")['X']
    data = jnp.array(data)
    [n, d] = data.shape
    m = 6
    keys = jax.random.split(key, 16)
    prob = MaF9(d=d, m=m)
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

def test_maf10():
    key = random.PRNGKey(1)
    data = jnp.load("data.npz")['X']
    data = jnp.array(data)
    [n, d] = data.shape
    m = 6
    keys = jax.random.split(key, 16)
    prob = MaF10(d=d, m=m)
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

def test_maf11():
    key = random.PRNGKey(1)
    data = jnp.load("data.npz")['X']
    data = jnp.array(data)
    [n, d] = data.shape
    m = 6
    keys = jax.random.split(key, 16)
    prob = MaF11(d=d, m=m)
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

def test_maf12():
    key = random.PRNGKey(1)
    data = jnp.load("data.npz")['X']
    data = jnp.array(data)
    [n, d] = data.shape
    m = 6
    keys = jax.random.split(key, 16)
    prob = MaF12(d=d, m=m)
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

def test_maf13():
    key = random.PRNGKey(1)
    data = jnp.load("data.npz")['X']
    data = jnp.array(data)
    [n, d] = data.shape
    m = 6
    keys = jax.random.split(key, 16)
    prob = MaF13(d=d, m=m)
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

def test_maf14():
    key = random.PRNGKey(1)
    data = jnp.load("data.npz")['X']
    data = jnp.array(data)
    [n, d] = data.shape
    m = 6
    keys = jax.random.split(key, 16)
    prob = MaF14(d=d, m=m)
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

def test_maf15():
    key = random.PRNGKey(1)
    data = jnp.load("data.npz")['X']
    data = jnp.array(data)
    [n, d] = data.shape
    m = 6
    keys = jax.random.split(key, 16)
    prob = MaF15(d=d, m=m)
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
