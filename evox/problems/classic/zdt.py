import jax
import jax.numpy as jnp
from functools import partial
import evox as ex
import chex


def _generic_zdt(f1, g, h, x):
    f1_x = f1(x)
    g_x = g(x)
    return jnp.array([f1_x, g_x * h(f1(x), g_x)])


@ex.jit_class
class ZDT(ex.Problem):
    def __init__(self, n, ref_num=100):
        self.n = n
        self._zdt = None
        self.ref_num = ref_num

    def evaluate(self, state: chex.PyTreeDef, X: jnp.ndarray):
        chex.assert_type(X, float)
        chex.assert_shape(X, (None, self.n))
        return state, jax.jit(jax.vmap(self._zdt))(X)

    def pf(self, state: chex.PyTreeDef):
        x = jnp.linspace(0, 1, self.ref_num)
        return state, jnp.c_[x, 1 - jnp.sqrt(x)]


class ZDT1(ZDT):
    def __init__(self, n):
        super().__init__(n)
        f1 = lambda x: x[0]
        g = lambda x: 1 + 9 * jnp.mean(x[1:])
        h = lambda f1, g: 1 - jnp.sqrt(f1 / g)
        self._zdt = partial(_generic_zdt, f1, g, h)



class ZDT2(ZDT):
    def __init__(self, n):
        super().__init__(n)
        f1 = lambda x: x[0]
        g = lambda x: 1 + 9 * jnp.mean(x[1:])
        h = lambda f1, g: 1 - (f1 / g) ** 2
        self._zdt = partial(_generic_zdt, f1, g, h)

    def pf(self, state: chex.PyTreeDef):
        x = jnp.linspace(0, 1, self.ref_num)
        return state, jnp.c_[x, 1 - x**2]



class ZDT3(ZDT):
    def __init__(self, n):
        super().__init__(n)
        f1 = lambda x: x[0]
        g = lambda x: 1 + 9 * jnp.mean(x[1:])
        h = lambda f1, g: 1 - jnp.sqrt(f1 / g) - (f1 / g) * jnp.sin(10 * jnp.pi * f1)
        self._zdt = partial(_generic_zdt, f1, g, h)

    def pf(self, state: chex.PyTreeDef):
        pass


class ZDT4(ZDT):
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

