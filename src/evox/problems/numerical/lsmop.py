import jax
import jax.numpy as jnp
import evox
import evox as ex
from src.evox.operators.sampling import UniformSampling
from src.evox.problems.numerical import Sphere as Sphere
import math
import chex


@evox.jit_class
class LSMOP(ex.Problem):
    """R. Cheng, Y. Jin, and M. Olhofer, Test problems for large-scale multiobjective and many-objective optimization, IEEE Transactions on Cybernetics, 2017, 47(12): 4108-4121."""

    def __init__(self, d=None, m=None, ref_num=1000):
        """init
        :param d: the dimension of decision space
        :param m: the number of object
        :param ref_num: ref_num * m is the Population of PF
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
        for i in range(1, self.m):
            c.append(3.8 * c[-1] * (1 - c[-1]))
        c = jnp.asarray(c)
        self.sublen = jnp.floor(c / jnp.sum(c) * self.d / self.nk)
        self.len_ = jnp.r_[0, jnp.cumsum(self.sublen * self.nk)]
        self.sublen = tuple(map(int, self.sublen))
        self.len_ = tuple(map(int, self.len_))

    def setup(self, key):
        return ex.State(key=key)

    """
       all LSMOPs using "for loop" is due to the two variables: self.sub and self.len_, which make dynamic slice and prevent the use of fori_loop.
    """

    def evaluate(self, state, X):
        return jax.jit(jax.vmap(self._lsmop))(X), state

    def pf(self, state):
        f = UniformSampling(self.ref_num * self.m, self.m)()[0] / 2
        return f, state

    @staticmethod
    def _Sphere(x):
        """get the sum of squares of each row in matrix x"""
        return jnp.sum(x**2, axis=1, keepdims=True)

    @staticmethod
    def _Giewank(x):
        f = (
            jnp.sum(x**2, axis=1, keepdims=True) / 4000
            - jnp.prod(
                jnp.cos(
                    x
                    / jnp.tile(
                        jnp.sqrt(jnp.arange(1, jnp.shape(x)[1] + 1)),
                        (jnp.shape(x)[0], 1),
                    )
                ),
                axis=1,
                keepdims=True,
            )
            + 1
        )
        return f

    @staticmethod
    def _Schwefel(x):
        return jnp.max(jnp.abs(x), keepdims=True, axis=1)

    @staticmethod
    def _Rastrigin(x):
        f = jnp.sum(x**2 - 10 * jnp.cos(2 * jnp.pi * x) + 10, axis=1, keepdims=True)
        return f

    @staticmethod
    def _Rosenbrock(x):
        f = jnp.sum(
            100 * ((x[:, : jnp.shape(x)[1] - 1]) ** 2 - x[:, 1 : jnp.shape(x)[1]]) ** 2
            + (x[:, : jnp.shape(x)[1] - 1] - 1) ** 2,
            axis=1,
            keepdims=True,
        )
        return f

    @staticmethod
    def _Ackley(x):
        f = (
            20
            - 20
            * jnp.exp(
                -0.2
                * jnp.sqrt(jnp.sum(x**2, axis=1, keepdims=True) / jnp.shape(x)[1])
            )
            - jnp.exp(
                jnp.sum(jnp.cos(2 * jnp.pi * x), axis=1, keepdims=True)
                / jnp.shape(x)[1]
            )
            + jnp.exp(1)
        )
        return f

    @staticmethod
    def _Griewank(x):
        f = (
            jnp.sum(x**2, axis=1, keepdims=True) / 4000
            - jnp.prod(
                jnp.cos(
                    x
                    / jnp.tile(
                        jnp.sqrt(jnp.arange(1, jnp.shape(x)[1] + 1)),
                        (jnp.shape(x)[0], 1),
                    )
                ),
                axis=1,
                keepdims=True,
            )
            + 1
        )
        return f


@evox.jit_class
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

    def evaluate(self, state, X):
        m = self.m
        n, d = jnp.shape(X)
        X = X.at[:, m - 1 : d].set(
            (1 + jnp.tile(jnp.arange(m, d + 1) / d, (n, 1))) * X[:, m - 1 : d]
            - jnp.tile(X[:, :1] * 10, (1, d - m + 1))
        )
        g = jnp.zeros([n, m])
        for i in range(0, m, 2):
            for j in range(1, self.nk + 1):
                g = g.at[:, i : i + 1].set(
                    g[:, i : i + 1]
                    + LSMOP._Sphere(
                        X[
                            :,
                            int(self.len_[i] + m - 1 + (j - 1) * self.sublen[i]) : int(
                                self.len_[i] + m - 1 + j * self.sublen[i]
                            ),
                        ],
                    )
                )
        for i in range(1, m, 2):
            for j in range(1, self.nk + 1):
                g = g.at[:, i : i + 1].set(
                    g[:, i : i + 1]
                    + LSMOP._Sphere(
                        X[
                            :,
                            int(self.len_[i] + m - 1 + (j - 1) * self.sublen[i]) : int(
                                self.len_[i] + m - 1 + j * self.sublen[i]
                            ),
                        ],
                    )
                )
        g = g / jnp.tile(jnp.array(self.sublen), (n, 1)) / self.nk
        f = (
            (1 + g)
            * jnp.fliplr(jnp.cumprod(jnp.c_[jnp.ones((n, 1)), X[:, : m - 1]], axis=1))
            * jnp.c_[jnp.ones((n, 1)), 1 - X[:, m - 2 :: -1]]
        )

        return f, state


@evox.jit_class
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

    def evaluate(self, state, X):
        n, d = jnp.shape(X)
        m = self.m
        X = X.at[:, m - 1 : d].set(
            (1 + jnp.tile(jnp.arange(m, d + 1) / d, (n, 1))) * X[:, m - 1 : d]
            - jnp.tile(X[:, :1] * 10, (1, d - m + 1))
        )
        g = jnp.zeros([n, m])
        for i in range(0, m, 2):
            for j in range(1, self.nk + 1):
                g = g.at[:, i : i + 1].set(
                    g[:, i : i + 1]
                    + LSMOP._Giewank(
                        X[
                            :,
                            int(self.len_[i] + m - 1 + (j - 1) * self.sublen[i]) : int(
                                self.len_[i] + m - 1 + j * self.sublen[i]
                            ),
                        ]
                    )
                )
        for i in range(1, m, 2):
            for j in range(1, self.nk + 1):
                g = g.at[:, i : i + 1].set(
                    g[:, i : i + 1]
                    + LSMOP._Schwefel(
                        X[
                            :,
                            int(self.len_[i] + m - 1 + (j - 1) * self.sublen[i]) : int(
                                self.len_[i] + m - 1 + j * self.sublen[i]
                            ),
                        ]
                    )
                )
        g = g / jnp.tile(jnp.array(self.sublen), (n, 1)) / self.nk
        f = (
            (1 + g)
            * jnp.fliplr(jnp.cumprod(jnp.c_[jnp.ones((n, 1)), X[:, : m - 1]], axis=1))
            * jnp.c_[jnp.ones((n, 1)), 1 - X[:, m - 2 :: -1]]
        )
        return f, state


@evox.jit_class
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

    def evaluate(self, state, X):
        n, d = jnp.shape(X)
        m = self.m
        X = X.at[:, m - 1 : d].set(
            (1 + jnp.tile(jnp.arange(m, d + 1) / d, (n, 1))) * X[:, m - 1 : d]
            - jnp.tile(X[:, :1] * 10, (1, d - m + 1))
        )
        g = jnp.zeros([n, m])
        for i in range(0, m, 2):
            for j in range(1, self.nk + 1):
                g = g.at[:, i : i + 1].set(
                    g[:, i : i + 1]
                    + LSMOP._Rastrigin(
                        X[
                            :,
                            int(self.len_[i] + m - 1 + (j - 1) * self.sublen[i]) : int(
                                self.len_[i] + m - 1 + j * self.sublen[i]
                            ),
                        ]
                    )
                )
        for i in range(1, m, 2):
            for j in range(1, self.nk + 1):
                g = g.at[:, i : i + 1].set(
                    g[:, i : i + 1]
                    + LSMOP._Rosenbrock(
                        X[
                            :,
                            int(self.len_[i] + m - 1 + (j - 1) * self.sublen[i]) : int(
                                self.len_[i] + m - 1 + j * self.sublen[i]
                            ),
                        ]
                    )
                )
        g = g / jnp.tile(jnp.array(self.sublen), (n, 1)) / self.nk
        f = (
            (1 + g)
            * jnp.fliplr(jnp.cumprod(jnp.c_[jnp.ones((n, 1)), X[:, : m - 1]], axis=1))
            * jnp.c_[jnp.ones((n, 1)), 1 - X[:, m - 2 :: -1]]
        )
        return f, state


@evox.jit_class
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

    def evaluate(self, state, X):
        n, d = jnp.shape(X)

        m = self.m
        X = X.at[:, m - 1 : d].set(
            (1 + jnp.tile(jnp.arange(m, d + 1) / d, (n, 1))) * X[:, m - 1 : d]
            - jnp.tile(X[:, :1] * 10, (1, d - m + 1))
        )
        g = jnp.zeros([n, m])
        for i in range(0, m, 2):
            for j in range(1, self.nk + 1):
                g = g.at[:, i : i + 1].set(
                    g[:, i : i + 1]
                    + self._Ackley(
                        X[
                            :,
                            int(self.len_[i] + m - 1 + (j - 1) * self.sublen[i]) : int(
                                self.len_[i] + m - 1 + j * self.sublen[i]
                            ),
                        ]
                    )
                )
        for i in range(1, m, 2):
            for j in range(1, self.nk + 1):
                g = g.at[:, i : i + 1].set(
                    g[:, i : i + 1]
                    + LSMOP._Griewank(
                        X[
                            :,
                            int(self.len_[i] + m - 1 + (j - 1) * self.sublen[i]) : int(
                                self.len_[i] + m - 1 + j * self.sublen[i]
                            ),
                        ]
                    )
                )
        g = g / jnp.tile(jnp.array(self.sublen), (n, 1)) / self.nk
        f = (
            (1 + g)
            * jnp.fliplr(jnp.cumprod(jnp.c_[jnp.ones((n, 1)), X[:, : m - 1]], axis=1))
            * jnp.c_[jnp.ones((n, 1)), 1 - X[:, m - 2 :: -1]]
        )
        return f, state


@evox.jit_class
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

    def evaluate(self, state, X):
        n, d = jnp.shape(X)
        m = self.m
        X = X.at[:, m - 1 : d].set(
            (1 + jnp.tile(jnp.cos(jnp.arange(m, d + 1) / d * jnp.pi / 2), (n, 1)))
            * X[:, m - 1 : d]
            - jnp.tile(X[:, :1] * 10, (1, d - m + 1))
        )
        g = jnp.zeros([n, m])
        for i in range(0, m, 2):
            for j in range(1, self.nk + 1):
                g = g.at[:, i : i + 1].set(
                    g[:, i : i + 1]
                    + LSMOP._Sphere(
                        X[
                            :,
                            int(self.len_[i] + m - 1 + (j - 1) * self.sublen[i]) : int(
                                self.len_[i] + m - 1 + j * self.sublen[i]
                            ),
                        ],
                    )
                )
        for i in range(1, m, 2):
            for j in range(1, self.nk + 1):
                g = g.at[:, i : i + 1].set(
                    g[:, i : i + 1]
                    + LSMOP._Sphere(
                        X[
                            :,
                            int(self.len_[i] + m - 1 + (j - 1) * self.sublen[i]) : int(
                                self.len_[i] + m - 1 + j * self.sublen[i]
                            ),
                        ],
                    )
                )
        g = g / jnp.tile(jnp.array(self.sublen), (n, 1)) / self.nk
        f = (
            (1 + g + jnp.c_[g[:, 1:], jnp.zeros((n, 1))])
            * jnp.fliplr(
                jnp.cumprod(
                    jnp.c_[jnp.ones((n, 1)), jnp.cos(X[:, : m - 1] * jnp.pi / 2)],
                    axis=1,
                )
            )
            * jnp.c_[jnp.ones((n, 1)), jnp.sin(X[:, m - 2 :: -1] * jnp.pi / 2)]
        )

        return f, state

    def pf(self, state):
        f = UniformSampling(self.ref_num * self.m, self.m)()[0] / 2
        return (
            f / jnp.tile(jnp.sqrt(jnp.sum(f**2, axis=1, keepdims=True)), (1, self.m)),
            state,
        )


@evox.jit_class
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

    def evaluate(self, state, X):
        n, d = jnp.shape(X)
        m = self.m
        X = X.at[:, m - 1 : d].set(
            (1 + jnp.tile(jnp.cos(jnp.arange(m, d + 1) / d * jnp.pi / 2), (n, 1)))
            * X[:, m - 1 : d]
            - jnp.tile(X[:, :1] * 10, (1, d - m + 1))
        )
        g = jnp.zeros((n, m))
        for i in range(0, m, 2):
            for j in range(1, self.nk + 1):
                g = g.at[:, i : i + 1].set(
                    g[:, i : i + 1]
                    + LSMOP._Rosenbrock(
                        X[
                            :,
                            int(self.len_[i] + m - 1 + (j - 1) * self.sublen[i]) : int(
                                self.len_[i] + m - 1 + j * self.sublen[i]
                            ),
                        ]
                    )
                )
        for i in range(1, m, 2):
            for j in range(1, self.nk + 1):
                g = g.at[:, i : i + 1].set(
                    g[:, i : i + 1]
                    + LSMOP._Schwefel(
                        X[
                            :,
                            int(self.len_[i] + m - 1 + (j - 1) * self.sublen[i]) : int(
                                self.len_[i] + m - 1 + j * self.sublen[i]
                            ),
                        ]
                    )
                )
        g = g / jnp.tile(jnp.array(self.sublen), (n, 1)) / self.nk
        f = (
            (1 + g + jnp.c_[g[:, 1:], jnp.zeros([n, 1])])
            * jnp.fliplr(
                jnp.cumprod(
                    jnp.c_[jnp.ones((n, 1)), jnp.cos(X[:, : m - 1] * jnp.pi / 2)],
                    axis=1,
                )
            )
            * jnp.c_[jnp.ones((n, 1)), jnp.sin(X[:, m - 2 :: -1] * jnp.pi / 2)]
        )
        return f, state


@evox.jit_class
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

    def evaluate(self, state, X):
        n, d = jnp.shape(X)
        m = self.m
        X = X.at[:, m - 1 : d].set(
            (1 + jnp.tile(jnp.cos(jnp.arange(m, d + 1) / d * jnp.pi / 2), (n, 1)))
            * X[:, m - 1 : d]
            - jnp.tile(X[:, :1] * 10, (1, d - m + 1))
        )
        g = jnp.zeros([n, m])
        for i in range(0, m, 2):
            for j in range(1, self.nk + 1):
                g = g.at[:, i : i + 1].set(
                    g[:, i : i + 1]
                    + LSMOP._Ackley(
                        X[
                            :,
                            int(self.len_[i] + m - 1 + (j - 1) * self.sublen[i]) : int(
                                self.len_[i] + m - 1 + j * self.sublen[i]
                            ),
                        ]
                    )
                )
        for i in range(1, m, 2):
            for j in range(1, self.nk + 1):
                g = g.at[:, i : i + 1].set(
                    g[:, i : i + 1]
                    + LSMOP._Rosenbrock(
                        X[
                            :,
                            int(self.len_[i] + m - 1 + (j - 1) * self.sublen[i]) : int(
                                self.len_[i] + m - 1 + j * self.sublen[i]
                            ),
                        ]
                    )
                )
        g = g / jnp.tile(jnp.array(self.sublen), (n, 1)) / self.nk
        f = (
            (1 + g + jnp.c_[g[:, 1:], jnp.zeros([n, 1])])
            * jnp.fliplr(
                jnp.cumprod(
                    jnp.c_[jnp.ones((n, 1)), jnp.cos(X[:, : m - 1] * jnp.pi / 2)],
                    axis=1,
                )
            )
            * jnp.c_[jnp.ones((n, 1)), jnp.sin(X[:, m - 2 :: -1] * jnp.pi / 2)]
        )
        return f, state

    def pf(self, state):
        f = UniformSampling(self.ref_num * self.m, self.m)()[0] / 2
        f = f / jnp.tile(jnp.sqrt(jnp.sum(f**2, axis=1, keepdims=True)), (1, self.m))
        return f, state


@evox.jit_class
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

    def evaluate(self, state, X):
        n, d = jnp.shape(X)
        m = self.m
        X = X.at[:, m - 1 : d].set(
            (1 + jnp.tile(jnp.cos(jnp.arange(m, d + 1) / d * jnp.pi / 2), (n, 1)))
            * X[:, m - 1 : d]
            - jnp.tile(X[:, :1] * 10, (1, d - m + 1))
        )
        g = jnp.zeros([n, m])
        for i in range(0, m, 2):
            for j in range(1, self.nk + 1):
                g = g.at[:, i : i + 1].set(
                    g[:, i : i + 1]
                    + LSMOP._Griewank(
                        X[
                            :,
                            int(self.len_[i] + m - 1 + (j - 1) * self.sublen[i]) : int(
                                self.len_[i] + m - 1 + j * self.sublen[i]
                            ),
                        ]
                    )
                )
        for i in range(1, m, 2):
            for j in range(1, self.nk + 1):
                g = g.at[:, i : i + 1].set(
                    g[:, i : i + 1]
                    + LSMOP._Sphere(
                        X[
                            :,
                            int(self.len_[i] + m - 1 + (j - 1) * self.sublen[i]) : int(
                                self.len_[i] + m - 1 + j * self.sublen[i]
                            ),
                        ]
                    )
                )
        g = g / jnp.tile(jnp.array(self.sublen), (n, 1)) / self.nk
        f = (
            (1 + g + jnp.c_[g[:, 1:], jnp.zeros([n, 1])])
            * jnp.fliplr(
                jnp.cumprod(
                    jnp.c_[jnp.ones((n, 1)), jnp.cos(X[:, : m - 1] * jnp.pi / 2)],
                    axis=1,
                )
            )
            * jnp.c_[jnp.ones((n, 1)), jnp.sin(X[:, m - 2 :: -1] * jnp.pi / 2)]
        )
        return f, state

    def pf(self, state: chex.PyTreeDef):
        f = UniformSampling(self.ref_num * self.m, self.m)()[0] / 2
        f = f / jnp.tile(jnp.sqrt(jnp.sum(f**2, axis=1, keepdims=True)), (1, self.m))
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

    @evox.jit_method
    def evaluate(self, state: chex.PyTreeDef, X):
        n, d = jnp.shape(X)
        self.N = n
        m = self.m
        X = X.at[:, m - 1 : d].set(
            (1 + jnp.tile(jnp.cos(jnp.arange(m, d + 1) / d * jnp.pi / 2), (n, 1)))
            * X[:, m - 1 : d]
            - jnp.tile(X[:, :1] * 10, (1, d - m + 1))
        )
        g = jnp.zeros((n, m))
        for i in range(0, m, 2):
            for j in range(1, self.nk + 1):
                g = g.at[:, i : i + 1].set(
                    g[:, i : i + 1]
                    + LSMOP._Sphere(
                        X[
                            :,
                            int(self.len_[i] + m - 1 + (j - 1) * self.sublen[i]) : int(
                                self.len_[i] + m - 1 + j * self.sublen[i]
                            ),
                        ]
                    )
                )
        for i in range(1, m, 2):
            for j in range(1, self.nk + 1):
                g = g.at[:, i : i + 1].set(
                    g[:, i : i + 1]
                    + LSMOP._Ackley(
                        X[
                            :,
                            int(self.len_[i] + m - 1 + (j - 1) * self.sublen[i]) : int(
                                self.len_[i] + m - 1 + j * self.sublen[i]
                            ),
                        ]
                    )
                )
        g = 1 + jnp.sum(
            g / jnp.tile(jnp.array(self.sublen), (n, 1)) / self.nk,
            axis=1,
            keepdims=True,
        )
        f = jnp.zeros((n, m))
        f = f.at[:, : m - 1].set(X[:, : m - 1])
        f = f.at[:, m - 1 : m].set(
            (1 + g)
            * (
                m
                - jnp.sum(
                    f[:, : m - 1]
                    / (1 + jnp.tile(g, (1, m - 1)))
                    * (1 + jnp.sin(3 * jnp.pi * f[:, : m - 1])),
                    axis=1,
                    keepdims=True,
                )
            )
        )
        return f, state

    def pf(self, state):
        interval = [0, 0.251412, 0.631627, 0.859401]
        median = (interval[1] - interval[0]) / (
            interval[3] - interval[2] + interval[1] - interval[0]
        )
        N = self.ref_num * self.m
        X = self._grid(N, self.m - 1)
        X = X.at[X <= median].set(
            X[X <= median] * (interval[1] - interval[0]) / median + interval[0]
        )
        X = X.at[X > median].set(
            (X[X > median] - median) * (interval[3] - interval[2]) / (1 - median)
            + interval[2]
        )
        p = jnp.c_[
            X,
            2
            * (
                self.m
                - jnp.sum(X / 2 * (1 + jnp.sin(3 * jnp.pi * X)), axis=1, keepdims=True)
            ),
        ]
        return p, state

    def _grid(self, N, M):
        gap = jnp.linspace(0, 1, int(math.ceil(N ** (1 / M))), dtype=jnp.float64)
        c = jnp.meshgrid(*([gap] * M))
        w = jnp.vstack([x.ravel() for x in c]).T
        return w
