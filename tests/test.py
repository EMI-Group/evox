from evox.operators.gaussian_processes import regression
import jax
import jax.numpy as jnp
import jax.random as jr
from gpjax.likelihoods import Gaussian
import gpjax as gpx
from evox.problems.numerical.maf import MaF3
import optax as ox

import jax.numpy as jnp
import jax

# 示例数据
n = 5  # 种群大小
K = 3  # 高斯模型数量
pop_dec = jnp.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])  # 种群决策变量
model = {
    "mean": jnp.array([[0, 0], [2, 2], [4, 4]]),
    "PI": jnp.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]])
}  # 高斯模型

# 使用循环的方法
distance_loop = jnp.zeros((n, K))
for k in range(K):
    diff = pop_dec - jnp.tile(model["mean"][k], (n, 1))
    distance_loop = distance_loop.at[:, k].set(
        jnp.sum(diff.dot(model["PI"][k]) * diff, axis=1)
    )

# 使用vmap的方法
def body_fun1(k):
    diff = pop_dec - jnp.tile(model["mean"][k], (n, 1))
    return jnp.sum(diff.dot(model["PI"][k]) * diff, axis=1)

def body_fun2(k):
    diff = pop_dec - jnp.tile(model["mean"][k], (n, 1))
    return jnp.sum(diff.dot(model["PI"][k]) * diff, axis=1)

distance_vmap = jax.vmap(body_fun1, in_axes=0)(jnp.arange(K))
distance_vmap2 = jax.vmap(body_fun2, out_axes=1)(jnp.arange(K))
print(distance_loop)
print(distance_vmap)
print(distance_vmap2)
