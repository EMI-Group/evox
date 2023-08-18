import jax
from matplotlib import path
import numpy as np
from matplotlib.path import Path
import jax.numpy as jnp


# 创建两个长度为 n 的向量
a = jnp.array([1, 2, 3])
b = jnp.array([4, 5, 6])
i = 1
# 使用 hstack 函数将它们水平堆叠
c = jnp.arange(i, i+2)

# 输出结果
print(c)
