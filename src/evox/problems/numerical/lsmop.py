import jax
import jax.numpy as jnp
import evox
import evox as ex
from src.evox.operators.sampling import UniformSampling
import chex
from functools import partial


@evox.jit_class
class LSMOP(ex.Problem):
    """LSMOP"""

    def __init__(self, d=None, m=None, ref_num=1000):
        """init
        :param d: the dimension of decision space
        :param m: the number of object
        :param ref_num: ???
        """
        self.nk = 5
        if m is None:
            self.m = 3
        else:
            self.m = m

        if d is None:
            self.d = self.m * 100
        else:
            self.d = d
        self._lsmop = None
        self.ref_num = ref_num
        # calculate the number of subgroup and their length
        c = [3.8 * 0.1 * (1 - 0.1)]
        tmp = [3.8 * 0.1 * (1 - 0.1)]
        for i in range(1, self.m):
            c.append(3.8 * c[- 1] * (1 - c[-1]))
        c = jnp.asarray(c)
        self.sublen = jnp.floor(c / jnp.sum(c) * self.d / self.nk)
        self.len_ = jnp.r_[0, jnp.cumsum(self.sublen * self.nk)]

    def setup(self, key: jax.Array):
        return ex.State(key=key)

    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
        chex.assert_type(X, float)
        chex.assert_shape(X, (None, self.n))
        return jax.jit(jax.vmap(self._lsmop))(X), state

    def pf(self, state: chex.PyTreeDef):
        f = UniformSampling(self.ref_num * self.m, self.m)()[0] / 2
        return f, state

    @staticmethod
    def _Sphere(x):
        """get the sum of squares of each row in matrix x"""
        return jnp.sum(x ** 2, axis=1, keepdims=True)

    @staticmethod
    def _Giewank(x):
        f = jnp.sum(x ** 2, axis=1, keepdims=True) / 4000 - jnp.prod(
            jnp.cos(x / jnp.tile(jnp.sqrt(jnp.arange(1, jnp.shape(x)[1] + 1)), (jnp.shape(x)[0], 1))), axis=1,
            keepdims=True) + 1
        return f

    @staticmethod
    def _Schwefel(x):
        return jnp.max(jnp.abs(x), keepdims=True, axis=1)

    @staticmethod
    def _Rastrigin(x):
        f = jnp.sum(x ** 2 - 10 * jnp.cos(2 * jnp.pi * x) + 10, axis=1, keepdims=True)
        return f

    @staticmethod
    def _Rosenbrock(x):
        f = jnp.sum(
            100 * ((x[:, :jnp.shape(x)[1] - 1]) ** 2 - x[:, 1:jnp.shape(x)[1]]) ** 2 + (
                    x[:, :jnp.shape(x)[1] - 1] - 1) ** 2,
            axis=1, keepdims=True)
        return f

    @staticmethod
    def _Ackley(x):
        # np.seterr(divide='ignore', invalid='ignore')
        f = 20 - 20 * jnp.exp(-0.2 * jnp.sqrt(jnp.sum(x ** 2, axis=1, keepdims=True) / jnp.shape(x)[1])) - jnp.exp(
            jnp.sum(jnp.cos(2 * jnp.pi * x), axis=1, keepdims=True) / jnp.shape(x)[1]) + jnp.exp(1)
        if jnp.isnan(f[0]):
            f = jnp.zeros([jnp.shape(f)[0], jnp.shape(f)[1]])
        return f

    @staticmethod
    def _Griewank(x):
        f = jnp.sum(x ** 2, axis=1, keepdims=True) / 4000 - jnp.prod(
            jnp.cos(x / jnp.tile(jnp.sqrt(jnp.arange(1, jnp.shape(x)[1] + 1)), (jnp.shape(x)[0], 1))), axis=1,
            keepdims=True) + 1
        return f


class LSMOP1(LSMOP):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)
        if m is None:
            self.m = 3
        else:
            self.m = m
        if d is None:
            self.d = self.m + 4
        else:
            self.d = d

    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
        m = self.m
        n, d = jnp.shape(X)
        # X[:, m - 1: d] = (1 + jnp.tile(jnp.arange(m, d + 1) / d, (n, 1))) * X[:, m - 1:d] - jnp.tile(X[:, :1] * 10, (1, d - m + 1))
        X = X.at[:, m - 1: d].set(
            (1 + jnp.tile(jnp.arange(m, d + 1) / d, (n, 1))) * X[:, m - 1:d] - jnp.tile(X[:, :1] * 10, (1, d - m + 1)))
        g = jnp.zeros([n, m])
        for i in range(0, m, 2):
            for j in range(1, self.nk + 1):
                temp = LSMOP._Sphere(X[:, int(self.len_[i] + m - 1 + (j - 1) * self.sublen[i]): int(
                    self.len_[i] + m - 1 + j * self.sublen[i])])
                g = g.at[:, i:i + 1].set(g[:, i:i + 1] + LSMOP._Sphere(X[:,
                                                                       int(self.len_[i] + m - 1 + (j - 1) * self.sublen[
                                                                           i]): int(
                                                                           self.len_[i] + m - 1 + j * self.sublen[i])]))
        for i in range(1, m, 2):
            for j in range(1, self.nk + 1):
                g = g.at[:, i:i + 1].set(g[:, i:i + 1] + LSMOP._Sphere(X[:,
                                                                       int(self.len_[i] + m - 1 + (j - 1) * self.sublen[
                                                                           i]): int(
                                                                           self.len_[i] + m - 1 + j * self.sublen[i])]))
        g = g / jnp.tile(self.sublen, (n, 1)) / self.nk
        f = (1 + g) * jnp.fliplr(jnp.cumprod(jnp.c_[jnp.ones((n, 1)), X[:, :m - 1]], axis=1)) * jnp.c_[
            jnp.ones((n, 1)), 1 - X[:, m - 2::-1]]

        return f, state


class LSMOP2(LSMOP):
    def __init__(self, d=None, m=None, ref_num=1000):
        if m is None:
            self.m = 3
        else:
            self.m = m
        if d is None:
            self.d = self.m + 4
        else:
            self.d = d
        super().__init__(d, m, ref_num)

    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
        n, d = jnp.shape(X)
        m = self.m
        X = X.at[:, m - 1: d].set((1 + jnp.tile(jnp.arange(m, d + 1) / d, (n, 1))) * X[:, m - 1:d] - jnp.tile(
            X[:, :1] * 10, (1, d - m + 1)))
        g = jnp.zeros([n, m])
        for i in range(0, m, 2):
            for j in range(1, self.nk + 1):
                g = g.at[:, i:i + 1].set(g[:, i:i + 1] + LSMOP._Giewank(X[:,
                                                                        int(self.len_[i] + m - 1 + (j - 1) *
                                                                            self.sublen[i]): int(
                                                                            self.len_[i] + m - 1 + j * self.sublen[
                                                                                i])]))
        for i in range(1, m, 2):
            for j in range(1, self.nk + 1):
                g = g.at[:, i:i + 1].set(g[:, i:i + 1] + LSMOP._Schwefel(X[:,
                                                                         int(self.len_[i] + m - 1 + (j - 1) *
                                                                             self.sublen[i]): int(
                                                                             self.len_[i] + m - 1 + j * self.sublen[
                                                                                 i])]))
        g = g / jnp.tile(self.sublen, (n, 1)) / self.nk
        f = (1 + g) * jnp.fliplr(jnp.cumprod(jnp.c_[jnp.ones((n, 1)), X[:, :m - 1]], axis=1)) * jnp.c_[
            jnp.ones((n, 1)), 1 - X[:, m - 2::-1]]
        return f, state


class LSMOP3(LSMOP):
    def __init__(self, d=None, m=None, ref_num=1000):
        if m is None:
            self.m = 3
        else:
            self.m = m
        if d is None:
            self.d = self.m + 4
        else:
            self.d = d
        super().__init__(d, m, ref_num)

    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
        n, d = jnp.shape(X)
        m = self.m
        X = X.at[:, m - 1: d].set((1 + jnp.tile(jnp.arange(m, d + 1) / d, (n, 1))) * X[:, m - 1:d] - jnp.tile(
            X[:, :1] * 10, (1, d - m + 1)))
        g = jnp.zeros([n, m])
        for i in range(0, m, 2):
            for j in range(1, self.nk + 1):
                g = g.at[:, i:i + 1].set(g[:, i:i + 1] + LSMOP._Rastrigin(X[:,
                                                                          int(self.len_[i] + m - 1 + (j - 1) *
                                                                              self.sublen[i]): int(
                                                                              self.len_[i] + m - 1 + j * self.sublen[
                                                                                  i])]))
        for i in range(1, m, 2):
            for j in range(1, self.nk + 1):
                g = g.at[:, i:i + 1].set(g[:, i:i + 1] + LSMOP._Rosenbrock(X[:,
                                                                           int(self.len_[i] + m - 1 + (j - 1) *
                                                                               self.sublen[i]): int(
                                                                               self.len_[i] + m - 1 + j * self.sublen[
                                                                                   i])]))
        g = g / jnp.tile(self.sublen, (n, 1)) / self.nk
        f = (1 + g) * jnp.fliplr(jnp.cumprod(jnp.c_[jnp.ones((n, 1)), X[:, :m - 1]], axis=1)) * jnp.c_[
            jnp.ones((n, 1)), 1 - X[:, m - 2::-1]]
        return f, state


class LSMOP4(LSMOP):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)
        if m is None:
            self.m = 3
        else:
            self.m = m
        if d is None:
            self.d = self.m + 4
        else:
            self.d = d

    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
        n, d = jnp.shape(X)
        m = self.m
        X = X.at[:, m - 1: d].set((1 + jnp.tile(jnp.arange(m, d + 1) / d, (n, 1))) * X[:, m - 1:d] - jnp.tile(
            X[:, :1] * 10, (1, d - m + 1)))
        g = jnp.zeros([n, m])
        for i in range(0, m, 2):
            for j in range(1, self.nk + 1):
                g = g.at[:, i:i + 1].set(g[:, i:i + 1] + LSMOP._Ackley(X[:,
                                                                       int(self.len_[i] + m - 1 + (j - 1) * self.sublen[
                                                                           i]): int(
                                                                           self.len_[i] + m - 1 + j * self.sublen[i])]))
        for i in range(1, m, 2):
            for j in range(1, self.nk + 1):
                g = g.at[:, i:i + 1].set(g[:, i:i + 1] + LSMOP._Griewank(X[:,
                                                                         int(self.len_[i] + m - 1 + (j - 1) *
                                                                             self.sublen[i]): int(
                                                                             self.len_[i] + m - 1 + j * self.sublen[
                                                                                 i])]))
        g = g / jnp.tile(self.sublen, (n, 1)) / self.nk
        f = (1 + g) * jnp.fliplr(jnp.cumprod(jnp.c_[jnp.ones((n, 1)), X[:, :m - 1]], axis=1)) * jnp.c_[
            jnp.ones((n, 1)), 1 - X[:, m - 2::-1]]
        return f, state


class LSMOP5(LSMOP):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)
        if m is None:
            self.m = 3
        else:
            self.m = m
        if d is None:
            self.d = self.m + 4
        else:
            self.d = d

    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
        n, d = jnp.shape(X)
        m = self.m
        X = X.at[:, m - 1: d].set((1 + jnp.tile(jnp.cos(jnp.arange(m, d + 1) / d * jnp.pi / 2), (n, 1))) * X[:,
                                                                                                           m - 1: d] - jnp.tile(
            X[:, :1] * 10, (1, d - m + 1)))
        g = jnp.zeros([n, m])
        for i in range(0, m, 2):
            for j in range(1, self.nk + 1):
                g = g.at[:, i:i + 1].set(g[:, i:i + 1] + LSMOP._Sphere(X[:,
                                                                       int(self.len_[i] + m - 1 + (j - 1) * self.sublen[
                                                                           i]): int(
                                                                           self.len_[i] + m - 1 + j * self.sublen[i])]))
        for i in range(1, m, 2):
            for j in range(1, self.nk + 1):
                g = g.at[:, i:i + 1].set(g[:, i:i + 1] + LSMOP._Sphere(X[:,
                                                                       int(self.len_[i] + m - 1 + (j - 1) * self.sublen[
                                                                           i]): int(
                                                                           self.len_[i] + m - 1 + j * self.sublen[i])]))
        g = g / jnp.tile(self.sublen, (n, 1)) / self.nk
        # index = jnp.where(self.sublen == 0)
        # if len(index[0]) != 0:
        #     g[:, index] = 0
        f = (1 + g + jnp.c_[g[:, 1:], jnp.zeros((n, 1))]) * jnp.fliplr(
            jnp.cumprod(jnp.c_[jnp.ones((n, 1)), jnp.cos(X[:, :m - 1] * jnp.pi / 2)], axis=1)) * jnp.c_[
                jnp.ones((n, 1)), jnp.sin(X[:, m - 2::-1] * jnp.pi / 2)]

        return f, state

    def pf(self, state: chex.PyTreeDef):
        f = UniformSampling(self.ref_num * self.m, self.m)()[0] / 2
        return f / jnp.tile(jnp.sqrt(jnp.sum(f ** 2, axis=1, keepdims=True)), (1, self.m)), state


class LSMOP6(LSMOP):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)
        if m is None:
            self.m = 3
        else:
            self.m = m
        if d is None:
            self.d = self.m + 4
        else:
            self.d = d

    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
        n, d = jnp.shape(X)
        m = self.m
        X = X.at[:, m - 1: d].set((1 + jnp.tile(jnp.cos(jnp.arange(m, d + 1) / d * jnp.pi / 2), (n, 1))) * X[:,
                                                                                                           m - 1: d] - jnp.tile(
            X[:, :1] * 10, (1, d - m + 1)))
        g = jnp.zeros((n, m))
        for i in range(0, m, 2):
            for j in range(1, self.nk + 1):
                g = g.at[:, i:i + 1].set(g[:, i:i + 1] + LSMOP._Rosenbrock(X[:,
                                                                           int(self.len_[i] + m - 1 + (j - 1) *
                                                                               self.sublen[i]): int(
                                                                               self.len_[i] + m - 1 + j * self.sublen[
                                                                                   i])]))
        for i in range(1, m, 2):
            for j in range(1, self.nk + 1):
                g = g.at[:, i:i + 1].set(g[:, i:i + 1] + LSMOP._Schwefel(X[:,
                                                                         int(self.len_[i] + m - 1 + (j - 1) *
                                                                             self.sublen[i]): int(
                                                                             self.len_[i] + m - 1 + j * self.sublen[
                                                                                 i])]))
        g = g / jnp.tile(self.sublen, (n, 1)) / self.nk
        f = (1 + g + jnp.c_[g[:, 1:], jnp.zeros([n, 1])]) * jnp.fliplr(
            jnp.cumprod(jnp.c_[jnp.ones((n, 1)), jnp.cos(X[:, :m - 1] * jnp.pi / 2)], axis=1)) * jnp.c_[
                jnp.ones((n, 1)), jnp.sin(X[:, m - 2::-1] * jnp.pi / 2)]
        return f, state


class LSMOP7(LSMOP):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)
        if m is None:
            self.m = 3
        else:
            self.m = m
        if d is None:
            self.d = self.m + 4
        else:
            self.d = d

    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
        n, d = jnp.shape(X)
        m = self.m
        X = X.at[:, m - 1: d].set((1 + jnp.tile(jnp.cos(jnp.arange(m, d + 1) / d * jnp.pi / 2), (n, 1))) * X[:,
                                                                                                           m - 1: d] - jnp.tile(
            X[:, :1] * 10, (1, d - m + 1)))
        g = jnp.zeros([n, m])
        for i in range(0, m, 2):
            for j in range(1, self.nk + 1):
                g = g.at[:, i:i + 1].set(g[:, i:i + 1] + LSMOP._Ackley(X[:,
                                                                       int(self.len_[i] + m - 1 + (j - 1) * self.sublen[
                                                                           i]): int(
                                                                           self.len_[i] + m - 1 + j * self.sublen[i])]))
        for i in range(1, m, 2):
            for j in range(1, self.nk + 1):
                g = g.at[:, i:i + 1].set(g[:, i:i + 1] + LSMOP._Rosenbrock(X[:,
                                                                           int(self.len_[i] + m - 1 + (j - 1) *
                                                                               self.sublen[i]): int(
                                                                               self.len_[i] + m - 1 + j * self.sublen[
                                                                                   i])]))
        g = g / jnp.tile(self.sublen, (n, 1)) / self.nk
        f = (1 + g + jnp.c_[g[:, 1:], jnp.zeros([n, 1])]) * jnp.fliplr(
            jnp.cumprod(jnp.c_[jnp.ones((n, 1)), jnp.cos(X[:, :m - 1] * jnp.pi / 2)], axis=1)) * jnp.c_[
                jnp.ones((n, 1)), jnp.sin(X[:, m - 2::-1] * jnp.pi / 2)]
        return f, state

    def pf(self, state: chex.PyTreeDef):
        f = UniformSampling(self.ref_num * self.m, self.m)()[0] / 2
        f = f / jnp.tile(jnp.sqrt(jnp.sum(f ** 2, axis=1, keepdims=True)), (1, self.m))
        return f, state


class LSMOP8(LSMOP):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)
        if m is None:
            self.m = 3
        else:
            self.m = m
        if d is None:
            self.d = self.m + 4
        else:
            self.d = d

    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
        n, d = jnp.shape(X)
        m = self.m
        X = X.at[:, m - 1: d].set((1 + jnp.tile(jnp.cos(jnp.arange(m, d + 1) / d * jnp.pi / 2), (n, 1))) * X[:,
                                                                                                           m - 1: d] - jnp.tile(
            X[:, :1] * 10, (1, d - m + 1)))
        g = jnp.zeros([n, m])
        for i in range(0, m, 2):
            for j in range(1, self.nk + 1):
                g = g.at[:, i:i + 1].set(g[:, i:i + 1] + LSMOP._Griewank(X[:,
                                                                         int(self.len_[i] + m - 1 + (j - 1) *
                                                                             self.sublen[i]): int(
                                                                             self.len_[i] + m - 1 + j * self.sublen[
                                                                                 i])]))
        for i in range(1, m, 2):
            for j in range(1, self.nk + 1):
                g = g.at[:, i:i + 1].set(g[:, i:i + 1] + LSMOP._Sphere(X[:,
                                                                       int(self.len_[i] + m - 1 + (j - 1) * self.sublen[
                                                                           i]): int(
                                                                           self.len_[i] + m - 1 + j * self.sublen[i])]))
        g = g / jnp.tile(self.sublen, (n, 1)) / self.nk
        f = (1 + g + jnp.c_[g[:, 1:], jnp.zeros([n, 1])]) * jnp.fliplr(
            jnp.cumprod(jnp.c_[jnp.ones((n, 1)), jnp.cos(X[:, :m - 1] * jnp.pi / 2)], axis=1)) * jnp.c_[
                jnp.ones((n, 1)), jnp.sin(X[:, m - 2::-1] * jnp.pi / 2)]
        return f, state

    def pf(self, state: chex.PyTreeDef):
        f = UniformSampling(self.ref_num * self.m, self.m)()[0] / 2
        f = f / jnp.tile(jnp.sqrt(jnp.sum(f ** 2, axis=1, keepdims=True)), (1, self.m))
        return f, state


class LSMOP9(LSMOP):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)
        if m is None:
            self.m = 3
        else:
            self.m = m
        if d is None:
            self.d = self.m + 4
        else:
            self.d = d

    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
        n, d = jnp.shape(X)
        self.N = n
        m = self.m
        # print(jnp.shape(jnp.tile(jnp.arange(m, d + 1), (n, 1)) * X[:, m - 1:d]))
        # print(jnp.shape(jnp.tile(X[:, :1] * 10, (1, d - m + 1))))
        # PopDec(:,M:D) = (1+repmat(cos((M:D)./D*pi/2),N,1)).*PopDec(:,M:D) - repmat(PopDec(:,1)*10,1,D-M+1);
        X = X.at[:, m - 1: d].set((1 + jnp.tile(jnp.cos(jnp.arange(m, d + 1) / d * jnp.pi / 2), (n, 1))) * X[:,
                                                                                                           m - 1: d] - jnp.tile(
            X[:, :1] * 10, (1, d - m + 1)))
        # X[:, m - 1: d] = (1 + jnp.tile(jnp.arange(m, d + 1) / d, (n, 1))) * X[:, m - 1:d] - jnp.tile(
        #     X[:, :1] * 10, (1, d - m + 1))
        g = jnp.zeros((n, m))
        for i in range(0, m, 2):
            for j in range(1, self.nk + 1):
                g = g.at[:, i:i + 1].set(g[:, i:i + 1] + LSMOP._Sphere(X[:,
                                                                       int(self.len_[i] + m - 1 + (j - 1) * self.sublen[
                                                                           i]): int(
                                                                           self.len_[i] + m - 1 + j * self.sublen[i])]))
        for i in range(1, m, 2):
            for j in range(1, self.nk + 1):
                g = g.at[:, i:i + 1].set(g[:, i:i + 1] + LSMOP._Ackley(X[:,
                                                                       int(self.len_[i] + m - 1 + (j - 1) * self.sublen[
                                                                           i]): int(
                                                                           self.len_[i] + m - 1 + j * self.sublen[i])]))
        # G = 1 + sum(G./repmat(obj.sublen,N,1)./obj.nk,2);
        g = 1 + jnp.sum(g / jnp.tile(self.sublen, (n, 1)) / self.nk, axis=1, keepdims=True)
        # f = jnp.random.random([temp, self.d])  # may have error
        f = jnp.zeros((n, m))
        # print("temp", temp)
        # print("D", self.d)
        # f[:, :m - 1] = X[:, :m - 1]
        # print("----22", f.shape)
        # print("b SHAPE", f[:, :m - 1].shape)
        f = f.at[:, :m - 1].set(X[:, :m - 1])
        # print(f[:, :m - 1].shape)
        # print((1 + jnp.tile(g, (1, m - 1))).shape)
        f = f.at[:, m - 1:m].set((1 + g) * (m - jnp.sum(
            f[:, :m - 1] / (1 + jnp.tile(g, (1, m - 1))) * (1 + jnp.sin(3 * jnp.pi * f[:, :m - 1])), axis=1,
            keepdims=True)))
        # print("X SHAOE", f[:, m-1:m].shape)
        return f, state

    def pf(self, state: chex.PyTreeDef):
        interval = [0, 0.251412, 0.631627, 0.859401]
        median = (interval[1] - interval[0]) / (interval[3] - interval[2] + interval[1] - interval[0])
        X = self._ReplicatePoint(self.N, self.m - 1)
        # print("x", X)
        X = X.at[X <= median].set(X[X <= median] * (interval[1] - interval[0]) / median + interval[0])
        X = X.at[X > median].set((X[X > median] - median) * (interval[3] - interval[2]) / (1 - median) + interval[2])
        p = jnp.c_[X, 2 * (self.m - jnp.sum(X / 2 * (1 + jnp.sin(3 * jnp.pi * X)), axis=1, keepdims=True))]
        return p, state

    def _ReplicatePoint(self, sample_num, M):
        if M > 1:
            sample_num = jnp.ceil(sample_num ** (1 / M)) ** M
            gap = jnp.arange(0, 1 + 1e-10, 1 / (sample_num ** (1 / M) - 1))
            length = len(gap)
            w = jnp.zeros((length ** M, M))
            s = "gap," * M
            statement = "jnp.meshgrid(" + s + ")"
            a = eval(statement)
            # print(a)
            w = w.at[:, 0].set(a[1][:].flatten('F'))
            w = w.at[:, 1].set(a[0][:].flatten('F'))
            for i in range(2, M):
                w = w.at[:, i].set(a[i][:].flatten('F'))

        else:
            w = (jnp.arange(0, 1 + 1e-10, 1 / (sample_num - 1))).transpose()
        return w
