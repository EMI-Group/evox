import jax.numpy as jnp
from jax import random
import sys
from matplotlib import path
import numpy as np
from matplotlib.path import Path
from lsmop import *

key = random.PRNGKey(1)
points = jax.random.uniform(key, shape=(4, 2))
n = 10
A = jnp.ones((4, 2))

print(jnp.hstack((points,A)))
print(jnp.hstack((points,A),axis=1))
# print(ND)


