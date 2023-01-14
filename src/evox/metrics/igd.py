import jax
import jax.numpy as jnp
from evox.utils import euclidean_dis


class IGD:
    def __init__(self, pf, objs):
        self.pf = pf
        self.objs = objs
        
    def calulate(self):
        return jnp.mean(jnp.amin(euclidean_dis(self.pf, self.objs), axis=1))