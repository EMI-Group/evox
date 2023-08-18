import jax
import jax.numpy as jnp
from jax import random
from jax import vmap
from matplotlib import path
import numpy as np
from matplotlib.path import Path

class MaF8():
    def __init__(self, d=None, m=None, ref_num=1000):
        if m is None:
            self.m = 3
        else:
            self.m = m
        if d is None:
            self.d = self.m + 4
        else:
            self.d = d
        self._maf = None
        self.ref_num = ref_num
        self.points = self.getPoints()


    def evaluate(self, X):
        f = self.eucl_dis(X, self.points)
        return f

    def pf(self):
        # 没有实现完，self.inpolygon 有问题
        n = 30
        temp = jnp.linspace(-1,1,num=jnp.ceil(jnp.sqrt(n)).astype(jnp.int32))
        x, y = jnp.meshgrid(temp, temp)
        ND   = self.inpolygon(x.reshape((-1,1)),y.reshape((-1,1)),self.points[:,0],self.points[:,1])
        f    = self.eucl_dis([x[ND],y[ND]], self.points);
        return f





    def eucl_dis(self,x,y):
        a_squared = jnp.sum(jnp.square(x), axis=1)
        b_squared = jnp.sum(jnp.square(y), axis=1)

        ab = jnp.dot(x, y.T)

        distance = jnp.sqrt(jnp.add(jnp.add(-2 * ab, a_squared[:, jnp.newaxis]), b_squared))
        return distance

    def _inpolygon(self, xq, yq, xv, yv):
        """
        wrong!
        """
        shape = xq.shape
        xq = xq.reshape(-1)
        yq = yq.reshape(-1)
        xv = xv.reshape(-1)
        yv = yv.reshape(-1)
        q = [(xq[i], yq[i]) for i in range(xq.shape[0])]
        p = path.Path([(xv[i], yv[i]) for i in range(xv.shape[0])])
        return p.contains_points(q).reshape(shape)


key = random.PRNGKey(1)
n = 30
d = 2
m = 6
i = 2
# x = random.normal(key,(n,d))
n = 30
temp = jnp.linspace(-1, 1, num=jnp.ceil(jnp.sqrt(n)).astype(jnp.int32))
x, y = jnp.meshgrid(temp, temp)
print(x.ravel(order="F").shape)
print(jnp.column_stack((x.ravel(order="F"),y.ravel(order="F"))).shape)
#
# poly_path = Path(self.points)
# ND = poly_path.contains_points()
#
# print(in_poly)  # [True, False]
