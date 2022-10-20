import evox as ex
import jax
import jax.numpy as jnp


@ex.jit_class
class SimulatedBinaryCrossover(ex.Operator):
    """Simulated binary crossover(SBX)

    Args:
        pro_c: the probabilities of doing crossover.
        dis_c: the distribution index of SBX.
        type: the type is that generate the number of individuals. Type 1 generate offspring of the same size as the parent. 
        Type 2 generate offspring of half of the population size of the parent. 
    """

    def __init__(self, pro_c=1, dis_c=20, type=1):
        self.pro_c = pro_c
        self.dis_c = dis_c
        self.type = type

    def setup(self, key):
        return ex.State(key=key)

    def __call__(self, state, x):
        key = state.key
        key, mu_key, beta1_key, beta2_key, beta3_key = jax.random.split(key, 5)
        n, _ = jnp.shape(x)
        parent1_dec = x[:n // 2, :]
        parent2_dec = x[n // 2:n // 2 * 2, :]
        n_p, d = jnp.shape(parent1_dec)
        beta = jnp.zeros((n_p, d))
        mu = jax.random.uniform(mu_key, shape=(n_p, d))
        beta = jnp.where(mu <= 0.5, jnp.power(
            2 * mu, 1 / (self.dis_c + 1)), beta)
        beta = jnp.where(mu > 0.5, jnp.power(
            2 - 2 * mu, -1 / (self.dis_c + 1)), beta)
        beta = beta * ((-1) ** jax.random.randint(beta1_key,
                       shape=(n_p, d), minval=0, maxval=2))
        beta = jnp.where(jax.random.uniform(
            beta2_key, shape=(n_p, d)) < 0.5, 1, beta)
        beta = jnp.where(jnp.tile(jax.random.uniform(
            beta3_key, shape=(n_p, 1)) > self.pro_c, (1, d)), 1, beta)
        if self.type == 1:
            offspring_dec = jnp.vstack(((parent1_dec + parent2_dec) / 2 + beta * (parent1_dec - parent2_dec) / 2,
                                        (parent1_dec + parent2_dec) / 2 - beta * (parent1_dec - parent2_dec) / 2))
            if n % 2 != 0:
                offspring_dec = jnp.r_[offspring_dec, x[-1:, :]]
        if self.type == 2:
            offspring_dec = (parent1_dec + parent2_dec) / 2 + \
                beta * (parent1_dec - parent2_dec) / 2
        return ex.State(key=key), offspring_dec
