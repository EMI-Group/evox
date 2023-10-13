import jax
import jax.numpy as jnp
from evox import Problem, State, jit_class
from evox.operators.sampling import UniformSampling, GridSampling


@jit_class
class DTLZTestSuit(Problem):
    """DTLZ

    link: https://link.springer.com/chapter/10.1007/1-84628-137-7_6
    """

    def __init__(self, d=None, m=None, ref_num=1000):
        self.d = d
        self.m = m
        self._dtlz = None
        self.ref_num = ref_num
        self.sample = UniformSampling(self.ref_num * self.m, self.m)

    def setup(self, key):
        return State(key=key)

    def evaluate(self, state, X):
        return jax.jit(jax.vmap(self._dtlz))(X), state

    def pf(self, state):
        f = self.sample()[0] / 2
        return f, state


class DTLZ1(DTLZTestSuit):
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

    def evaluate(self, state, X):
        m = self.m
        n, d = jnp.shape(X)

        g = 100 * (
            d
            - m
            + 1
            + jnp.sum(
                (X[:, m - 1 :] - 0.5) ** 2
                - jnp.cos(20 * jnp.pi * (X[:, m - 1 :] - 0.5)),
                axis=1,
                keepdims=True,
            )
        )
        f = (
            0.5
            * jnp.tile(1 + g, (1, m))
            * jnp.fliplr(jnp.cumprod(jnp.c_[jnp.ones((n, 1)), X[:, : m - 1]], axis=1))
            * jnp.c_[jnp.ones((n, 1)), 1 - X[:, m - 2 :: -1]]
        )
        return f, state


class DTLZ2(DTLZTestSuit):
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

    def evaluate(self, state, X):
        m = self.m
        g = jnp.sum((X[:, m - 1 :] - 0.5) ** 2, axis=1, keepdims=True)
        f = (
            jnp.tile(1 + g, (1, m))
            * jnp.fliplr(
                jnp.cumprod(
                    jnp.c_[
                        jnp.ones((jnp.shape(g)[0], 1)),
                        jnp.maximum(jnp.cos(X[:, : m - 1] * jnp.pi / 2), 0),
                    ],
                    axis=1,
                )
            )
            * jnp.c_[
                jnp.ones((jnp.shape(g)[0], 1)), jnp.sin(X[:, m - 2 :: -1] * jnp.pi / 2)
            ]
        )

        return f, state

    def pf(self, state):
        f = self.sample()[0]
        f /= jnp.tile(jnp.sqrt(jnp.sum(f**2, axis=1, keepdims=True)), (1, self.m))
        return f, state


class DTLZ3(DTLZ2):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, state, X):
        n, d = jnp.shape(X)
        m = self.m
        g = 100 * (
            d
            - m
            + 1
            + jnp.sum(
                (
                    (X[:, m - 1 :] - 0.5) ** 2
                    - jnp.cos(20 * jnp.pi * (X[:, m - 1 :] - 0.5))
                ),
                axis=1,
                keepdims=True,
            )
        )
        f = (
            jnp.tile(1 + g, (1, m))
            * jnp.fliplr(
                jnp.cumprod(
                    jnp.c_[
                        jnp.ones((n, 1)),
                        jnp.maximum(jnp.cos(X[:, : m - 1] * jnp.pi / 2), 0),
                    ],
                    axis=1,
                )
            )
            * jnp.c_[jnp.ones((n, 1)), jnp.sin(X[:, m - 2 :: -1] * jnp.pi / 2)]
        )

        return f, state


class DTLZ4(DTLZ2):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, state, X):
        m = self.m
        X = X.at[:, : m - 1].power(100)
        g = jnp.sum((X[:, m - 1 :] - 0.5) ** 2, axis=1, keepdims=True)
        f = (
            jnp.tile(1 + g, (1, m))
            * jnp.fliplr(
                jnp.cumprod(
                    jnp.c_[
                        jnp.ones((jnp.shape(g)[0], 1)),
                        jnp.maximum(jnp.cos(X[:, : m - 1] * jnp.pi / 2), 0),
                    ],
                    axis=1,
                )
            )
            * jnp.c_[
                jnp.ones((jnp.shape(g)[0], 1)), jnp.sin(X[:, m - 2 :: -1] * jnp.pi / 2)
            ]
        )

        return f, state


class DTLZ5(DTLZTestSuit):
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

    def evaluate(self, state, X):
        m = self.m
        g = jnp.sum((X[:, m - 1 :] - 0.5) ** 2, axis=1, keepdims=True)
        temp = jnp.tile(g, (1, m - 2))
        X = X.at[:, 1 : m - 1].set((1 + 2 * temp * X[:, 1 : m - 1]) / (2 + 2 * temp))
        f = (
            jnp.tile(1 + g, (1, m))
            * jnp.fliplr(
                jnp.cumprod(
                    jnp.c_[
                        jnp.ones((jnp.shape(g)[0], 1)),
                        jnp.maximum(jnp.cos(X[:, : m - 1] * jnp.pi / 2), 0),
                    ],
                    axis=1,
                )
            )
            * jnp.c_[
                jnp.ones((jnp.shape(g)[0], 1)), jnp.sin(X[:, m - 2 :: -1] * jnp.pi / 2)
            ]
        )
        return f, state

    def pf(self, state):
        n = self.ref_num * self.m
        f = jnp.vstack(
            (
                jnp.hstack(((jnp.arange(0, 1, 1.0 / (n - 1))), 1.0)),
                jnp.hstack(((jnp.arange(1, 0, -1.0 / (n - 1))), 0.0)),
            )
        ).T
        f /= jnp.tile(
            jnp.sqrt(jnp.sum(f**2, axis=1, keepdims=True)), (1, jnp.shape(f)[1])
        )

        for i in range(self.m - 2):
            f = jnp.c_[f[:, 0], f]

        f = (
            f
            / jnp.sqrt(2)
            * jnp.tile(
                jnp.hstack((self.m - 2, jnp.arange(self.m - 2, -1, -1))),
                (jnp.shape(f)[0], 1),
            )
        )
        return f, state


class DTLZ6(DTLZTestSuit):
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

    def evaluate(self, state, X):
        m = self.m
        g = jnp.sum((X[:, m - 1 :] ** 0.1), axis=1, keepdims=True)
        temp = jnp.tile(g, (1, m - 2))
        X = X.at[:, 1 : m - 1].set((1 + 2 * temp * X[:, 1 : m - 1]) / (2 + 2 * temp))

        f = (
            jnp.tile(1 + g, (1, m))
            * jnp.fliplr(
                jnp.cumprod(
                    jnp.c_[
                        jnp.ones((jnp.shape(g)[0], 1)),
                        jnp.maximum(jnp.cos(X[:, : m - 1] * jnp.pi / 2), 0),
                    ],
                    axis=1,
                )
            )
            * jnp.c_[
                jnp.ones((jnp.shape(g)[0], 1)), jnp.sin(X[:, m - 2 :: -1] * jnp.pi / 2)
            ]
        )
        return f, state

    def pf(self, state):
        n = self.ref_num * self.m
        f = jnp.vstack(
            (
                jnp.hstack(((jnp.arange(0, 1, 1.0 / (n - 1))), 1.0)),
                jnp.hstack(((jnp.arange(1, 0, -1.0 / (n - 1))), 0.0)),
            )
        ).T
        f /= jnp.tile(
            jnp.sqrt(jnp.sum(f**2, axis=1, keepdims=True)), (1, jnp.shape(f)[1])
        )

        for i in range(self.m - 2):
            f = jnp.c_[f[:, 0], f]

        f = (
            f
            / jnp.sqrt(2)
            * jnp.tile(
                jnp.hstack((self.m - 2, jnp.arange(self.m - 2, -1, -1))),
                (jnp.shape(f)[0], 1),
            )
        )
        return f, state


class DTLZ7(DTLZTestSuit):
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
        self.sample = GridSampling(self.ref_num * self.m, self.m - 1)

    def evaluate(self, state, X):
        n, d = jnp.shape(X)
        m = self.m
        f = jnp.zeros((n, m))
        g = 1 + 9 * jnp.mean(X[:, m - 1 :], axis=1, keepdims=True)
        f = f.at[:, : m - 1].set(X[:, : m - 1])
        f = f.at[:, m - 1 :].set(
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
        interval = jnp.array([0, 0.251412, 0.631627, 0.859401])
        median = (interval[1] - interval[0]) / (
            interval[3] - interval[2] + interval[1] - interval[0]
        )

        x = self.sample()[0]

        mask_less_equal_median = x <= median
        mask_greater_median = x > median

        x = jnp.where(
            mask_less_equal_median,
            x * (interval[1] - interval[0]) / median + interval[0],
            x,
        )
        x = jnp.where(
            mask_greater_median,
            (x - median) * (interval[3] - interval[2]) / (1 - median) + interval[2],
            x,
        )

        last_col = 2 * (
            self.m
            - jnp.sum(x / 2 * (1 + jnp.sin(3 * jnp.pi * x)), axis=1, keepdims=True)
        )

        pf = jnp.hstack([x, last_col])
        return pf, state
