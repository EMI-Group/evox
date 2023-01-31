import jax
import jax.numpy as jnp


def alpha(n, k):
    alpha = jnp.zeros(n+1)
        
    for i in range(1, n + 1):
        alpha = alpha.at[i].set(jnp.prod(jnp.array([(k - j) / (n - j) for j in range(1, i)])) / i)

    return alpha
    

class HyperVolume:
    def __init__(self, ref, objs, sample_num=100000):
        self.ref = ref
        self.objs = objs
        self.sample_num = sample_num
        
    def calulate(self, key):
        key, subkey = jax.random.split(key)
        n, m = jnp.shape(self.objs)
        max_value = self.ref
        min_value = jnp.min(self.objs, axis=0)
        v = jnp.prod(max_value - min_value)
        s = jax.random.uniform(subkey, shape=(self.sample_num, m), minval=min_value, maxval=max_value)
        
        dom = jax.vmap(lambda x, y: jnp.all(x <= y, axis=1), in_axes=(0, None))(self.objs, s)
        n_dom = jnp.sum(dom, axis=0)
        
        hv = self._hv_monte_carlo(dom, v, n_dom=n_dom)
        return hv

    def _hv_monte_carlo(self, dom, v, n_dom=None):
        n, sample_num = jnp.shape(dom)
        if n_dom is None:
            n_dom = jnp.sum(dom, axis=0)
        
        a = alpha(n, n)
        hv = jnp.sum(jnp.array([a[n_dom[dom[i]]].sum(axis=0) for i in range(n)])) / sample_num * v
        return hv
    
    

    