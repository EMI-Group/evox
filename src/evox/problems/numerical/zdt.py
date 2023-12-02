import jax
import jax.numpy as jnp
from functools import partial
from evox import jit_class, Problem


def _generic_zdt(f1, g, h, x):
    f1_x = f1(x)
    g_x = g(x)
    return jnp.array([f1_x, g_x * h(f1(x), g_x)])


@jit_class
class ZDTTestSuit(Problem):
    def __init__(self, n, ref_num=100):
        self.n = n
        self._zdt = None
        self.ref_num = ref_num

    def evaluate(self, state, X: jax.Array):
        return jax.jit(jax.vmap(self._zdt))(X), state

    def pf(self, state):
        x = jnp.linspace(0, 1, self.ref_num)
        return jnp.c_[x, 1 - jnp.sqrt(x)], state


class ZDT1(ZDTTestSuit):
    def __init__(self, n):
        super().__init__(n)
        f1 = lambda x: x[0]
        g = lambda x: 1 + 9 * jnp.mean(x[1:])
        h = lambda f1, g: 1 - jnp.sqrt(f1 / g)
        self._zdt = partial(_generic_zdt, f1, g, h)


class ZDT2(ZDTTestSuit):
    def __init__(self, n):
        super().__init__(n)
        f1 = lambda x: x[0]
        g = lambda x: 1 + 9 * jnp.mean(x[1:])
        h = lambda f1, g: 1 - (f1 / g) ** 2
        self._zdt = partial(_generic_zdt, f1, g, h)

    def pf(self, state):
        x = jnp.linspace(0, 1, self.ref_num)
        return jnp.c_[x, 1 - x**2], state


class ZDT3(ZDTTestSuit):
    def __init__(self, n):
        super().__init__(n)
        f1 = lambda x: x[0]
        g = lambda x: 1 + 9 * jnp.mean(x[1:])
        h = lambda f1, g: 1 - jnp.sqrt(f1 / g) - (f1 / g) * jnp.sin(10 * jnp.pi * f1)
        self._zdt = partial(_generic_zdt, f1, g, h)

    def pf(self, state):
        r = jnp.array([[0, 0.0830], [0.1822, 0.2577], [0.4093, 0.4538], [0.6183, 0.6525], [0.8233, 0.8518]])

        f1 = jnp.linspace(r[:, 0], r[:, 1], int(self.ref_num / len(r)))
        f2 = 1 - jnp.sqrt(f1) - f1 * jnp.sin(10 * jnp.pi * f1)
        pf = jnp.array([f1, f2]).T
        pf = jnp.row_stack(pf)
        return pf, state


class ZDT4(ZDTTestSuit):
    def __init__(self, n):
        super().__init__(n)
        f1 = lambda x: x[0]
        g = (
            lambda x: 1
            + 10 * (self.n - 1)
            + jnp.sum(x[1:] ** 2 - 10 * jnp.cos(4 * jnp.pi * x[1:]))
        )
        h = lambda f1, g: 1 - jnp.sqrt(f1 / g)
        self._zdt = partial(_generic_zdt, f1, g, h)


class ZDT6(ZDTTestSuit):
    def __init__(self, n):
        super().__init__(n)
        f1 = lambda x: 1 - jnp.exp(-4 * x[0]) * jnp.sin(6 * jnp.pi * x[0])**6
        g = lambda x: 1 + 9 * (jnp.sum(x[1:]) / 9)**0.25
        h = lambda f1, g: 1 - (f1 / g)**2
        self._zdt = partial(_generic_zdt, f1, g, h)

    def pf(self, state):
        min_f1 = 0.280775
        f1 = jnp.linspace(min_f1, 1, self.ref_num)
        return jnp.c_[f1, 1 - f1**2], state
