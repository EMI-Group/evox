import jax
import jax.numpy as jnp
from evox import Problem, State, jit_class
from evox.operators.sampling import UniformSampling, LatinHypercubeSampling
import chex
import pkgutil, json


@jit_class
class DTLZTestSuit(Problem):
    """DTLZ"""

    def __init__(self, d=None, m=None, ref_num=1000):
        self.d = d
        self.m = m
        self._dtlz = None
        self.ref_num = ref_num
        self.sample = UniformSampling(self.ref_num * self.m, self.m)

    def setup(self, key):
        return State(key=key)

    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
        chex.assert_type(X, float)
        chex.assert_shape(X, (None, self.d))
        return jax.jit(jax.vmap(self._dtlz))(X), state

    def pf(self, state: chex.PyTreeDef):
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

    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
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

    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
        m = self.m
        g = jnp.sum((X[:, m - 1 :] - 0.5) ** 2, axis=1, keepdims=True)
        f = (
            jnp.tile(1 + g, (1, m))
            * jnp.fliplr(
                jnp.cumprod(
                    jnp.c_[
                        jnp.ones((jnp.shape(g)[0], 1)),
                        jnp.cos(X[:, : m - 1] * jnp.pi / 2),
                    ],
                    axis=1,
                )
            )
            * jnp.c_[
                jnp.ones((jnp.shape(g)[0], 1)), jnp.sin(X[:, m - 2 :: -1] * jnp.pi / 2)
            ]
        )

        return f, state

    def pf(self, state: chex.PyTreeDef):
        f = self.sample()[0]
        f /= jnp.tile(jnp.sqrt(jnp.sum(f**2, axis=1, keepdims=True)), (1, self.m))
        return f, state


class DTLZ3(DTLZTestSuit):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
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
                    jnp.c_[jnp.ones((n, 1)), jnp.cos(X[:, : m - 1] * jnp.pi / 2)],
                    axis=1,
                )
            )
            * jnp.c_[jnp.ones((n, 1)), jnp.sin(X[:, m - 2 :: -1] * jnp.pi / 2)]
        )

        return f, state


class DTLZ4(DTLZTestSuit):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
        m = self.m
        X = X.at[:, : m - 1].power(100)
        g = jnp.sum((X[:, m - 1 :] - 0.5) ** 2, axis=1, keepdims=True)
        f = (
            jnp.tile(1 + g, (1, m))
            * jnp.fliplr(
                jnp.cumprod(
                    jnp.c_[
                        jnp.ones((jnp.shape(g)[0], 1)),
                        jnp.cos(X[:, : m - 1] * jnp.pi / 2),
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

    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
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
                        jnp.cos(X[:, : m - 1] * jnp.pi / 2),
                    ],
                    axis=1,
                )
            )
            * jnp.c_[
                jnp.ones((jnp.shape(g)[0], 1)), jnp.sin(X[:, m - 2 :: -1] * jnp.pi / 2)
            ]
        )
        return f, state

    def pf(self, state: chex.PyTreeDef):
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

    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
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
                        jnp.cos(X[:, : m - 1] * jnp.pi / 2),
                    ],
                    axis=1,
                )
            )
            * jnp.c_[
                jnp.ones((jnp.shape(g)[0], 1)), jnp.sin(X[:, m - 2 :: -1] * jnp.pi / 2)
            ]
        )
        return f, state

    def pf(self, state: chex.PyTreeDef):
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

    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
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

    def pf(self, state: chex.PyTreeDef):
        data_bytes = pkgutil.get_data(__name__, "data/dtlz7_pf.json")
        data = data_bytes.decode()
        pf = json.loads(data)
        return jnp.array(pf), state
