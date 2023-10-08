import jax
import jax.numpy as jnp
import evox
from evox import Problem, State
from evox.operators.sampling import UniformSampling
from evox.problems.numerical import Sphere, Griewank, Rosenbrock, Ackley
import math


@evox.jit_class
class LSMOP(Problem):
    """R. Cheng, Y. Jin, and M. Olhofer, Test problems for large-scale multiobjective and many-objective optimization, IEEE Transactions on Cybernetics, 2017, 47(12): 4108-4121."""

    def __init__(self, d=None, m=None, ref_num=1000):
        """init
        :param d: the dimension of decision space
        :param m: the number of object
        :param ref_num: ref_num * m is the Population of PF
        """
        super().__init__()
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
        self.len = jnp.r_[0, jnp.cumsum(self.sublen * self.nk)]
        self.sublen = tuple(map(int, self.sublen))
        self.len = tuple(map(int, self.len))
        self.sphere = Sphere()
        self.griewank = Griewank()
        self.rosenbrock = Rosenbrock()
        self.ackley = Ackley()

    def setup(self, key):
        return State(key=key)

    def evaluate(self, state, X):
        return jax.jit(jax.vmap(self._lsmop))(X), state

    def pf(self, state):
        f = UniformSampling(self.ref_num * self.m, self.m)()[0] / 2
        return f, state

    """
       it is totally different with schwefel_func in cec2022_so.py
    """

    @staticmethod
    def _Schwefel(state, x):
        return jnp.max(jnp.abs(x), axis=-1), state

    """
        there is a little difference between with rastrigin_func in cec2022_so.py
    """

    @staticmethod
    def _Rastrigin(state, x):
        f = jnp.sum(x**2 - 10 * jnp.cos(2 * jnp.pi * x) + 10, axis=1)
        return f, state

    def _loop(
        self,
        state,
        out_start,
        out_end,
        outer_loop_type,
        inner_start,
        inner_end,
        inner_fun1,
        inner_fun2,
        x,
        g,
    ):
        def inner_loop(i, inner_fun, g):
            for j in range(inner_start, inner_end):
                start = self.len[i] + self.m - 1 + j * self.sublen[i]
                end = start + self.sublen[i]
                temp = x[:, start:end]
                g = g.at[:, i].set(g[:, i] + inner_fun(state, temp)[0])
            return g

        if outer_loop_type == "all":
            for i in range(out_start, out_end):
                g = inner_loop(i, inner_fun1, g)
        else:
            for i in range(out_start, out_end, 2):
                g = inner_loop(i, inner_fun1, g)
            for i in range(out_start + 1, out_end, 2):
                g = inner_loop(i, inner_fun2, g)
        return g


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
        g = self._loop(state, 0, m, "all", 0, self.nk, self.sphere.evaluate, None, X, g)
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
        g = self._loop(
            state,
            0,
            m,
            "jump",
            0,
            self.nk,
            self.griewank.evaluate,
            LSMOP._Schwefel,
            X,
            g,
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
        g = self._loop(
            state,
            0,
            m,
            "jump",
            0,
            self.nk,
            LSMOP._Rastrigin,
            self.rosenbrock.evaluate,
            X,
            g,
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
        g = self._loop(
            state,
            0,
            m,
            "jump",
            0,
            self.nk,
            self.ackley.evaluate,
            self.griewank.evaluate,
            X,
            g,
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
        g = self._loop(state, 0, m, "all", 0, self.nk, self.sphere.evaluate, None, X, g)
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
        g = self._loop(
            state,
            0,
            m,
            "jump",
            0,
            self.nk,
            self.rosenbrock.evaluate,
            LSMOP._Schwefel,
            X,
            g,
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
        g = self._loop(
            state,
            0,
            m,
            "jump",
            0,
            self.nk,
            self.ackley.evaluate,
            self.rosenbrock.evaluate,
            X,
            g,
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
        g = self._loop(
            state,
            0,
            m,
            "jump",
            0,
            self.nk,
            self.griewank.evaluate,
            self.sphere.evaluate,
            X,
            g,
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
    def evaluate(self, state, X):
        n, d = jnp.shape(X)
        self.N = n
        m = self.m
        X = X.at[:, m - 1 : d].set(
            (1 + jnp.tile(jnp.cos(jnp.arange(m, d + 1) / d * jnp.pi / 2), (n, 1)))
            * X[:, m - 1 : d]
            - jnp.tile(X[:, :1] * 10, (1, d - m + 1))
        )
        g = jnp.zeros((n, m))
        g = self._loop(
            state,
            0,
            m,
            "jump",
            0,
            self.nk,
            self.sphere.evaluate,
            self.ackley.evaluate,
            X,
            g,
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
