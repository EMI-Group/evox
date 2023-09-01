import jax
import jax.numpy as jnp
from jax import random
from jax import vmap
from matplotlib import path
import numpy as np
from matplotlib.path import Path
from maf import *
import chex
import pytest

key = random.PRNGKey(1)
data = jnp.load("data.npz")['X']
[n, d] = data.shape
m = 3
# key = jax.random.PRNGKey(12345)
keys = jax.random.split(key, 16)

maf = MaF3(d=d, m=m)
state = maf.init(keys)
# state = maf.setup(keys)

f, new_state = maf.evaluate(state, data)
print(f)


def pf(self):
    # n = self.ref_num * self.m
    n = 1000
    r = UniformSampling(n, self.m)()[0]
    c = jnp.zeros((n, self.m - 1))
    for i in range(1, n + 1):
        for j in range(2, self.m + 1):
            temp = r[i - 1, j - 1] / r[i - 1, 0] * jnp.prod(c[i - 1, self.m - j + 1:self.m - 1])
            c = c.at[i - 1, self.m - j - 1].set(jnp.sqrt(1 / (1 + temp ** 2)))
    if self.m > 5:
        c = c * (jnp.cos(jnp.pi / 8) - jnp.cos(3 * jnp.pi / 8)) + jnp.cos(3 * jnp.pi / 8)
    else:
        # temp = jnp.any(jnp.logical_or(c < jnp.cos(3 * jnp.pi / 8), c > jnp.cos(jnp.pi / 8)), axis=1)
        # c = jnp.delete(c, temp.flatten() == 1, axis=0)
        # 有问题，c被清空了
        c = c[jnp.all((c >= jnp.cos(3 * jnp.pi / 8)) & (c <= jnp.cos(jnp.pi / 8)), axis=1)]

    n, _ = jnp.shape(c)
    f = jnp.fliplr(jnp.cumprod(jnp.hstack([jnp.ones((n, 1)), c[:, :self.m - 1]]), axis=1)) * jnp.hstack(
        [jnp.ones((n, 1)), jnp.sqrt(1 - c[:, self.m - 2::-1] ** 2)])
    return f