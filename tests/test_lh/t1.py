import jax
import jax.numpy as jnp
from jax import random
from maf import *

key = random.PRNGKey(1)
data = jnp.load("data.npz")['X']
data = jnp.array(data)
[n, d] = data.shape
m = 6
# key = jax.random.PRNGKey(12345)
keys = jax.random.split(key, 16)

maf = MaF10(d=d, m=m)
state = maf.init(keys)
# state = maf.setup(keys)

f, new_state = maf.evaluate(state, data)
# f, new_state = maf.pf(state)
print(f.shape)
print(f)

# maf10 还没过，t4有问题