import jax
import jax.numpy as jnp
import evox as ex
from evox.operators.sampling import UniformSampling, LatinHypercubeSampling
import chex
from functools import partial


class DTLZ(ex.Problem):
    """DTLZ"""

    def __init__(self, d=None, m=None, ref_num=1000,):
        self.d = d
        self.m = m
        self._dtlz = None
        self.ref_num = ref_num
        
    def setup(self, key):
        return ex.State(key=key)

    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
        chex.assert_type(X, float)
        chex.assert_shape(X, (None, self.n))
        return state, jax.jit(jax.vmap(self._dtlz))(X)

    def pf(self, state: chex.PyTreeDef):
        f = UniformSampling(self.ref_num * self.m, self.m).random()[0] / 2
        # f = LatinHypercubeSampling(self.ref_num * self.m, self.m).random(state.key)[0] / 2
        return state, f


class DTLZ1(DTLZ):
    def __init__(self, d=None, m=None, ref_num=100):
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
        m = self.m
        n, d = jnp.shape(X)

        g = 100 * (d - m + 1 + jnp.sum(
            (X[:, m - 1:] - 0.5) ** 2 -
            jnp.cos(20 * jnp.pi * (X[:, m - 1:] - 0.5)),
            axis=1, keepdims=True))
        f = 0.5 * jnp.tile(1 + g, (1, m)) * jnp.fliplr(jnp.cumprod(
            jnp.c_[jnp.ones((n, 1)), X[:, :m - 1]], axis=1)) * \
            jnp.c_[jnp.ones((n, 1)), 1 - X[:, m - 2::-1]]
        return state, f


class DTLZ2(DTLZ):
    def __init__(self, d=None, m=None, ref_num=1000):
        if m is None:
            self.m = 3
        else:
            self.m = m
        if d is None:
            self.d = self.m + 9
        else:
            self.d = d
        super().__init__(d, m, ref_num)

    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
        m = self.m
        g = jnp.sum((X[:, m - 1:] - 0.5) ** 2, axis=1, keepdims=True)
        f = jnp.tile(1 + g, (1, m)) * jnp.fliplr(jnp.cumprod(jnp.c_[jnp.ones((jnp.shape(g)[0], 1)),
                                                                    jnp.cos(X[:, :m - 1] * jnp.pi / 2)], axis=1)) * jnp.c_[jnp.ones((jnp.shape(g)[0], 1)), jnp.sin(
                                                                        X[:, m - 2::-1] * jnp.pi / 2)]

        return state, f

    def pf(self, state: chex.PyTreeDef):
        f = UniformSampling(self.ref_num * self.m, self.m).random()[0]
        f /= jnp.tile(jnp.sqrt(jnp.sum(f ** 2, axis=1,
                      keepdims=True)), (1, self.m))
        return state, f


class DTLZ3(DTLZ2):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
        n, d = jnp.shape(X)
        m = self.m
        g = 100 * (d - m + 1 + jnp.sum(
            ((X[:, m - 1:] - 0.5) ** 2 -
             jnp.cos(20 * jnp.pi * (X[:, m - 1:] - 0.5))),
            axis=1, keepdims=True))
        f = jnp.tile(1 + g, (1, m)) * jnp.fliplr(jnp.cumprod(jnp.c_[jnp.ones((n, 1)),
                                                                    jnp.cos(X[:, :m - 1] * jnp.pi / 2)],
                                                             axis=1)) * \
            jnp.c_[jnp.ones((n, 1)), jnp.sin(X[:, m - 2::-1] * jnp.pi / 2)]

        return state, f


class DTLZ4(DTLZ2):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
        m = self.m
        X = X.at[:, :m - 1].power(100)
        g = jnp.sum((X[:, m - 1:] - 0.5) ** 2, axis=1, keepdims=True)
        f = jnp.tile(1 + g, (1, m)) * jnp.fliplr(jnp.cumprod(jnp.c_[jnp.ones((jnp.shape(g)[0], 1)),
                                                                    jnp.cos(X[:, :m - 1] * jnp.pi / 2)],
                                                             axis=1)) * \
            jnp.c_[jnp.ones((jnp.shape(g)[0], 1)), jnp.sin(
                X[:, m - 2::-1] * jnp.pi / 2)]

        return state, f


class DTLZ5(DTLZ):
    def __init__(self, d=None, m=None, ref_num=1000):
        if m is None:
            self.m = 3
        else:
            self.m = m

        if d is None:
            self.d = self.m + 9
        else:
            self.d = d
        super().__init__(d, m, ref_num)

    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
        m = self.m
        g = jnp.sum((X[:, m - 1:] - 0.5) ** 2, axis=1, keepdims=True)
        temp = jnp.tile(g, (1, m - 2))
        X = X.at[:, 1:m - 1].set((1 + 2 * temp *
                                 X[:, 1:m - 1]) / (2 + 2 * temp))
        f = jnp.tile(1 + g, (1, m)) * jnp.fliplr(jnp.cumprod(jnp.c_[jnp.ones((jnp.shape(g)[0], 1)),
                                                                    jnp.cos(X[:, :m - 1] * jnp.pi / 2)],
                                                             axis=1)) * \
            jnp.c_[jnp.ones((jnp.shape(g)[0], 1)), jnp.sin(
                X[:, m - 2::-1] * jnp.pi / 2)]
        return state, f

    def pf(self, state: chex.PyTreeDef):
        n = self.ref_num * self.m
        f = jnp.vstack((jnp.hstack(((jnp.arange(0, 1, 1. / (n - 1))), 1.)),
                       jnp.hstack(((jnp.arange(1, 0, -1. / (n - 1))), 0.)))).T
        f /= jnp.tile(jnp.sqrt(jnp.sum(f ** 2, axis=1,
                      keepdims=True)), (1, jnp.shape(f)[1]))

        for i in range(self.m - 2):
            f = jnp.c_[f[:, 0], f]

        f = f / jnp.sqrt(2) * jnp.tile(jnp.hstack((self.m - 2,
                                                   jnp.arange(self.m - 2, -1, -1))), (jnp.shape(f)[0], 1))
        return state, f


class DTLZ6(DTLZ):
    def __init__(self, d=None, m=None, ref_num=1000):
        if m is None:
            self.m = 3
        else:
            self.m = m
        if d is None:
            self.d = self.m + 9
        else:
            self.d = d
        super().__init__(d, m, ref_num)

    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
        m = self.m
        g = jnp.sum((X[:, m - 1:] ** 0.1), axis=1, keepdims=True)
        temp = jnp.tile(g, (1, m - 2))
        X = X.at[:, 1:m - 1].set((1 + 2 * temp *
                                 X[:, 1:m - 1]) / (2 + 2 * temp))

        f = jnp.tile(1 + g, (1, m)) * jnp.fliplr(jnp.cumprod(jnp.c_[jnp.ones((jnp.shape(g)[0], 1)),
                                                                    jnp.cos(X[:, :m - 1] * jnp.pi / 2)],
                                                             axis=1)) * \
            jnp.c_[jnp.ones((jnp.shape(g)[0], 1)), jnp.sin(
                X[:, m - 2::-1] * jnp.pi / 2)]
        return state, f

    def pf(self, state: chex.PyTreeDef):
        n = self.ref_num * self.m
        f = jnp.vstack((jnp.hstack(((jnp.arange(0, 1, 1. / (n - 1))), 1.)),
                       jnp.hstack(((jnp.arange(1, 0, -1. / (n - 1))), 0.)))).T
        f /= jnp.tile(jnp.sqrt(jnp.sum(f ** 2, axis=1,
                      keepdims=True)), (1, jnp.shape(f)[1]))

        for i in range(self.m - 2):
            f = jnp.c_[f[:, 0], f]

        f = f / jnp.sqrt(2) * jnp.tile(jnp.hstack((self.m - 2,
                                                   jnp.arange(self.m - 2, -1, -1))), (jnp.shape(f)[0], 1))
        return state, f


class DTLZ7(DTLZ):
    def __init__(self, d=None, m=None, ref_num=1000):
        if m is None:
            self.m = 3
        else:
            self.m = m
        if d is None:
            self.d = self.m + 19
        else:
            self.d = d
        super().__init__(d, m, ref_num)

    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
        n, d = jnp.shape(X)
        m = self.m
        f = jnp.zeros((n, m))
        g = 1 + jnp.mean(X[:, m-1:], axis=1, keepdims=True)
        f = f.at[:, m-1].set((1 + g) * (m - jnp.sum(X[:, :m-1] / (1 + jnp.tile(g, (1, m-1)))
                                                    * (1 + jnp.sin(3 * jnp.pi * X[:, :m-1])), axis=1, keepdims=True)))

        return state, f

    def pf(self, state: chex.PyTreeDef):
        intervel = jnp.array([0, 0.251412, 0.631627, 0.859401])
        median = (intervel[1] - intervel[0]) / (intervel[3] -
                                                intervel[2] + intervel[1] - intervel[0])
        # ReplicatePoint function
        return super().pf(state)


class DTLZ8(DTLZ):
    def __init__(self, d=None, m=None, ref_num=1000):
        if m is None:
            self.m = 2
        else:
            self.m = m

        if d is None:
            self.d = self.m*10
        else:
            self.d = d
        super().__init__(d, m, ref_num)

    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
        n, d = jnp.shape(X)
        m = self.m
        f = jnp.zeros((n, m))
        for i in range(m):
            f = f.at[:, i].set(
                jnp.mean(X[:, int(i*d/m):int((i+1)*d/m)], axis=1, keepdims=True))

        return state, f

    def pf(self, state: chex.PyTreeDef):
        return super().pf(state)


class DTLZ9(DTLZ):
    def __init__(self, d=None, m=None, ref_num=1000):
        if m is None:
            self.m = 2
        else:
            self.m = m

        if d is None:
            self.d = self.m*10
        else:
            self.d = d
        self.d = int(jnp.ceil(self.d/self.m)*self.m)
        super().__init__(d, m, ref_num)

    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
        n, d = jnp.shape(X)
        X = X**0.1
        m = self.m
        f = jnp.zeros([n, m])
        for i in range(m):
            f = f.at[:, i].set(jnp.sum(X[:, int(i*d/m):int((i+1)*d/m)],
                                       axis=1, keepdims=True))
        return state, f

    def pf(self, state: chex.PyTreeDef):
        n = self.ref_num * self.m
        temp = jnp.vstack(
            (jnp.hstack(((jnp.arange(0, 1, 1. / (n - 1))), 1.)))).T
        f = jnp.c_[jnp.tile(jnp.cos(0.5*jnp.pi*temp),
                            (1, self.m-1)), jnp.sin(0.5*jnp.pi*temp)]
        return state, f
