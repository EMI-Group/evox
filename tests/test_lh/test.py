import jax.numpy as jnp
import jax.random
from jax import random
import sys
from matplotlib import path
import numpy as np
from matplotlib.path import Path
from lsmop import *



# key = random.PRNGKey(1)
# a = jax.random.uniform(key, shape=(3,))
# X = jax.random.uniform(jax.random.PRNGKey(0), (3, 2))
# Y = jax.random.uniform(jax.random.PRNGKey(1), (3, 2))
A = jnp.array([0, 1, 0, 1, 1])
B = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
mask = A != 0
e = B.at[mask,:].set(0)
C = jnp.arange(mask.shape[0])[(mask == 0).nonzero()[0]]
print(e)



