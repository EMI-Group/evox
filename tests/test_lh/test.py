import jax.numpy as jnp
import jax.random
from jax import random
import sys
from matplotlib import path
import numpy as np
from matplotlib.path import Path
from lsmop import *

def pdist2(X, Y):
    # 计算X和Y之间的欧氏距离矩阵
    dist_matrix = jnp.linalg.norm(X[:, None] - Y, axis=-1)
    return dist_matrix

key = random.PRNGKey(1)
# a = jax.random.uniform(key, shape=(3,))
# X = jax.random.uniform(jax.random.PRNGKey(0), (3, 2))
# Y = jax.random.uniform(jax.random.PRNGKey(1), (3, 2))
a = jnp.array([1,2,3])
b = jnp.array([[2, 3],
               [5, 6],
              [8, 9]])
# c = a + b
# # a = b[:, jnp.ones((2)).astype(int)]
#
# d = pdist2(b, b)

# print(a.shape)
# print(a)
# print(b.shape)
# print(b)
print(jnp.maximum(a,3))



