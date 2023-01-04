import evox as ex
import jax
import jax.numpy as jnp


@ex.jit_class
class PmMutation(ex.Operator):
    """Polynomial mutation

    Args:
        pro_m: the expectation of number of bits doing mutation.
        dis_m: the distribution index of polynomial mutation.
    """

    def __init__(self, pro_m=1, dis_m=20):
        self.pro_m = pro_m
        self.dis_m = dis_m

    def setup(self, key):
        return ex.State(key=key)

    def __call__(self, state, x, boundary=None):
        key, subkey1, subkey2 = jax.random.split(state.key, 3)
        if jnp.shape(x)[0] == 1:
            pop_dec = x
        else:
            pop_dec = x[:(len(x)//2)*2, :]
        n, d = jnp.shape(pop_dec)
        site = jax.random.uniform(subkey1, shape=(n, d)) < self.pro_m / d
        mu = jax.random.uniform(subkey2, shape=(n, d))

        temp = site & (mu <= 0.5)
        lower, upper = jnp.tile(
            boundary[0], (n, 1)), jnp.tile(boundary[1], (n, 1))
        pop_dec = jnp.maximum(jnp.minimum(pop_dec, upper), lower)
        norm = jnp.where(temp, (pop_dec-lower)/(upper-lower), 0)
        pop_dec = jnp.where(temp, pop_dec+(upper - lower) *
                            (jnp.power(2. * mu + (1. - 2. * mu) * jnp.power(1. - norm, self.dis_m + 1.),
                                       1. / (self.dis_m + 1)) - 1.), pop_dec)

        temp = site & (mu > 0.5)
        norm = jnp.where(temp, (upper-pop_dec) / (upper - lower), 0)
        pop_dec = jnp.where(temp, pop_dec+(upper - lower) *
                            (1. - jnp.power(
                                2. * (1. - mu) + 2. * (mu - 0.5) *
                                jnp.power(1. - norm, self.dis_m + 1.),
                                1. / (self.dis_m + 1.))), pop_dec)
        if jnp.shape(x)[0] % 2 != 0:
            pop_dec = jnp.r_[pop_dec, x[-1:, :]]

        return ex.State(key=key), pop_dec
