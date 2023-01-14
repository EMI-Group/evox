import jax
import jax.numpy as jnp


def alpha(n, k):
    alpha = jnp.zeros(n+1)
        
    for i in range(1, n + 1):
        # alpha[i] = jnp.prod(jnp.array([(k - j) / (N - j) for j in range(1, i)])) / i
        alpha = alpha.at[i].set(jnp.prod(jnp.array([(k - j) / (n - j) for j in range(1, i)])) / i)
    
    
    # def out_body(i, out_val):
        
    #     def inner_body(j, inner_val):
    #         return 0
    #     prod = jax.lax.fori_loop()
    #     out_val = out_val.at[i].set(prod / i)
    #     return out_val
    
    # alpha = jax.lax.fori_loop(1, n+1, out_body, alpha)

    return alpha
    

# @jax.jit
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
        
        # dom = jnp.array([jnp.all(self.objs[i] <= s, axis=1) for i in range(n)])
        dom = jax.vmap(lambda x, y: jnp.all(x <= y, axis=1), in_axes=(0, None))(self.objs, s)
        # print(jnp.shape(dom1))
        # print(dom1)
        # print(jnp.all(dom1==dom))
        # dom2 = dom.at[0, 0].set(True)
        # print(jnp.all(dom1==dom2))
        n_dom = jnp.sum(dom, axis=0)
        
        hv = self._hv_monte_carlo(dom, v, n_dom=n_dom)
        return hv

    def _hv_monte_carlo(self, dom, v, n_dom=None):
        n, sample_num = jnp.shape(dom)
        if n_dom is None:
            n_dom = jnp.sum(dom, axis=0)
        
        a = alpha(n, n)
        # hit = jax.vmap(lambda x:jnp.sum(a[n_dom[x]], axis=0), in_axes=0)(dom)
        # hv1 = jnp.sum(hit, axis=0) / sample_num * v
        # print("hv1", hv1)
        hv = jnp.sum(jnp.array([a[n_dom[dom[i]]].sum(axis=0) for i in range(n)])) / sample_num * v
        return hv
    
    

    