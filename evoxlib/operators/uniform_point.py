import jax
import jax.numpy as jnp
from jax import lax
from itertools import combinations as n_choose_k
from jax.scipy.special import gammaln
from jax._src.lax.lax import _const as _lax_const


def comb(N,k):
    """Calculate the combinations

    """    
    one = _lax_const(N, 1)
    N_plus_1 = lax.add(N,one)
    k_plus_1 = lax.add(k,one)
    return jnp.rint(lax.exp(lax.sub(gammaln(N_plus_1), lax.add(gammaln(k_plus_1), gammaln(lax.sub(N_plus_1,k))))))


def uniform_point(n, m):
    """Generate uniformly distributed points on the hyperplane.

    Args:
        n (int): the population size.
        m (int): the number of objective.

    Returns:
        w: weight vector.
        n: the size of weight vector.
    """    
    h1 = 1
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

