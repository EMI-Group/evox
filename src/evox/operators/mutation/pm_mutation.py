from evox import jit_class
from jax import jit, random
import jax.numpy as jnp


@jit
def polynomial(key, x, boundary, pro_m, dis_m):
    subkey1, subkey2 = random.split(key)
    if jnp.shape(x)[0] == 1:
        pop_dec = x
    else:
        pop_dec = x[: (len(x) // 2) * 2, :]
    n, d = jnp.shape(pop_dec)
    site = random.uniform(subkey1, shape=(n, d)) < pro_m / d
    mu = random.uniform(subkey2, shape=(n, d))

    temp = site & (mu <= 0.5)
    lower, upper = jnp.tile(boundary[0], (n, 1)), jnp.tile(boundary[1], (n, 1))
    pop_dec = jnp.maximum(jnp.minimum(pop_dec, upper), lower)
    norm = jnp.where(temp, (pop_dec - lower) / (upper - lower), 0)
    pop_dec = jnp.where(
        temp,
        pop_dec
        + (upper - lower)
        * (
            jnp.power(
                2.0 * mu + (1.0 - 2.0 * mu) * jnp.power(1.0 - norm, dis_m + 1.0),
                1.0 / (dis_m + 1),
            )
            - 1.0
        ),
        pop_dec,
    )

    temp = site & (mu > 0.5)
    norm = jnp.where(temp, (upper - pop_dec) / (upper - lower), 0)
    pop_dec = jnp.where(
        temp,
        pop_dec
        + (upper - lower)
        * (
            1.0
            - jnp.power(
                2.0 * (1.0 - mu)
                + 2.0 * (mu - 0.5) * jnp.power(1.0 - norm, dis_m + 1.0),
                1.0 / (dis_m + 1.0),
            )
        ),
        pop_dec,
    )
    if jnp.shape(x)[0] % 2 != 0:
        pop_dec = jnp.r_[pop_dec, x[-1:, :]]

    return pop_dec


@jit_class
class Polynomial:
    """Polynomial mutation
    Inspired by PlatEMO.

    Args:
        pro_m: the expectation of number of bits doing mutation.
        dis_m: the distribution index of polynomial mutation.
    """

    def __init__(self, boundary, pro_m=1, dis_m=20):
        self.boundary = boundary
        self.pro_m = pro_m
        self.dis_m = dis_m

    def __call__(self, key, x):
        return polynomial(key, x, self.boundary, self.pro_m, self.dis_m)
