import jax.numpy as jnp
from jax import random
import sys
from matplotlib import path
import numpy as np
from matplotlib.path import Path
from lsmop import *

key = random.PRNGKey(1)
points = jax.random.uniform(key, shape=(4, 2))
n = 1000
m = 6
r = UniformSampling(n, m)()[0]

print(r)
# print(jnp.hstack((points,A),axis=1))


