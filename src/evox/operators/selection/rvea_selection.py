import evox as ex
import jax
import jax.numpy as jnp
import jax.experimental.host_callback as hcb
from evox.utils import cos_dist


@ex.jit_class
class ReferenceVectorGuidedSelection(ex.Operator):
    """Reference vector guided environmental selection.

    """    
    
    def __init__(self, x=None, v=None, theta=None):
        self.x = x
        self.v = v
        self.theta = theta

    def setup(self, key):
        return ex.State(key=key)

    def __call__(self, state, x, v, theta):
        self.x = x
        self.v = v
        self.theta = theta
        key, subkey = jax.random.split(state.key)
        n, m = jnp.shape(self.x)
        nv = jnp.shape(self.v)[0]
        obj = self.x

        obj -= jnp.tile(jnp.min(obj, axis=0), (n, 1))

        cosine = cos_dist(self.v, self.v)
        cosine = jnp.where(jnp.eye(jnp.shape(cosine)[0], dtype=bool), 0, cosine) 
        gamma = jnp.min(jnp.arccos(cosine), axis=1)

        angle = jnp.arccos(cos_dist(obj, self.v))

        associate = jnp.argmin(angle, axis=1)

        next_ind = jnp.full(nv, -1)
        is_null = jnp.sum(next_ind)
        
        
        def update_next(i, sub_index ,next_ind):
            apd = (1+m*theta*angle[sub_index, i]/gamma[i]) * jnp.sqrt(jnp.sum(obj[sub_index, :]**2, axis=1))
            apd_max = jnp.max(apd)
            noise = jnp.where(sub_index==-1, apd_max, 0)
            best = jnp.argmin(apd+noise)
            next_ind = next_ind.at[i].set(sub_index[best.astype(int)])
            return next_ind
    
        def no_update(i, sub_index ,next_ind):
            return next_ind
        
        def body_fun(i, val):
            sub_index = jnp.where(associate == i, size=nv, fill_value=-1)[0] 
            next_ind = jax.lax.cond(jnp.sum(sub_index) != is_null, update_next, no_update, i, sub_index, val)
            return next_ind
        
        next_ind = jax.lax.fori_loop(0, nv, body_fun, next_ind)

        return ex.State(key=key), next_ind
    
    

