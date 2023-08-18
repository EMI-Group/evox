import jax
import jax.numpy as jnp
from jax import random
from jax import vmap
from matplotlib import path
import numpy as np
from matplotlib.path import Path


key = random.PRNGKey(1)
i = 1
K = 2
M = 3
x = random.normal(key,(10,20))
n = 30
temp = jnp.tile(i, (n,2))
a = x[1, jnp.arange(int((i - 2) * K / (M - 1)), int((i - 1) * K / (M - 1)))]
b = x[1, jnp.arange(int((i-1)*K/(M-1)),int(i*K/(M-1)))]

print(a)
print(b)
# print(temp.shape)
#
# poly_path = Path(self.points)
# ND = poly_path.contains_points()
#
# print(in_poly)  # [True, False]
