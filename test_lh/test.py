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
A = jnp.ones((n - 1))
c = jnp.minimum(0,A)
print(A)
print(c)
# print(ND)


