import jax
import jax.numpy as jnp
from functools import partial
import evoxlib as exl
import chex

@exl.jit_class
class DTLZ(exl.Problem):
    def __init__(self n, ref_num=10000,):
        self.n = n
        self._dtlz = None
        self.ref_num = ref_num

    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
        chex.assert_type(X, float)
        chex.assert_shape(X, (None, self.n))
        return state, jax.jit(jax.vmap(self._dtlz))(X)
    
    def pf(self)
        pass