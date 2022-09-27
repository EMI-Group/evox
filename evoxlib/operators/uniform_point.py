import jax
import jax.numpy as jnp
# import math
from scipy.special import comb
from itertools import combinations as n_choose_k
from functools import partial

# @partial(jax.jit, static_argnums=2)
def uniform_point(n, m):
    h1 = 1
    # print(math.comb(h1 + m, m - 1))
    while comb(h1 + m, m - 1) <= n:
        h1 += 1
    w = jnp.array(list(n_choose_k(range(1, h1 + m), m-1))) - \
        jnp.tile(jnp.array(range(m-1)), (comb(h1+m-1, m-1).astype(int), 1)) - 1
    w = (jnp.c_[w, jnp.zeros((jnp.shape(w)[0], 1)) + h1] -
         jnp.c_[jnp.zeros((jnp.shape(w)[0], 1)), w]) / h1
    if h1 < m:
        h2 = 0
        while comb(h1+m-1, m-1) + comb(h2+m, m-1) <= n:
            h2 += 1
        if h2 > 0:
            w2 = jnp.array(list(n_choose_k(range(1, h2+m), m-1))) - \
                 jnp.tile(jnp.array(range(m - 1)), (comb(h2+m-1, m-1).astype(int), 1)) - 1
            w2 = (jnp.c_[w2, jnp.zeros((jnp.shape(w2)[0], 1))+h2] -
                  jnp.c_[jnp.zeros((jnp.shape(w2)[0], 1)), w2]) / h2
            w = jnp.r_[w, w2/2. + 1./(2.*m)]
    w = jnp.maximum(w, 1e-6)
    n = jnp.shape(w)[0]
    return w, n

# @jax.jit
# def pf(x):
#     x = jnp.asarray(x)
#     f = jnp.asarray(uniform_point(275, x)[0])
#     return f

if __name__ == '__main__':
    w, n = uniform_point(275, 10)
    print(w.dtype)
    print(n)
