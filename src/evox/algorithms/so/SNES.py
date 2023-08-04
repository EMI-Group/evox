
import jax
import jax.numpy as jnp
import evox
from flax import linen as nn

def get_recombination_weights(popsize: int, use_baseline: bool = True):
    def get_weight(i):
        return jnp.maximum(0, jnp.log(popsize / 2 + 1) - jnp.log(i))
    weights = jax.vmap(get_weight)(jnp.arange(1, popsize + 1))
    weights_norm = weights / jnp.sum(weights)
    return weights_norm - use_baseline * (1 / popsize)

def get_temp_weights(popsize: int, temperature: float):
    ranks = jnp.arange(popsize)
    ranks /= ranks.size - 1
    ranks = ranks - 0.5
    weights = nn.softmax(-temperature * ranks)
    return weights

@evox.jit_class
class SNES(evox.Algorithm):                                     
    def __init__(self,
                 pop_size: int,
                 center_init: jax.Array,
                 sigma: float = 1.0,

                lrate_mean: float = 1.0,
                temperature: float = 0.0,
                init_min: float = 0.0,
                init_max: float = 0.0,
                ):
        
        super().__init__()

        self.num_dims = center_init.shape[0]
        self.center_init = center_init
        self.popsize = pop_size
        self.sigma = sigma

        self.lrate_mean=lrate_mean
        self.lrate_sigma=(3 + jnp.log(self.num_dims)) / (
            5 * jnp.sqrt(self.num_dims)
        )
        self.temperature = temperature
        self.init_max=init_max
        self.init_min=init_min
    
    def setup(self, key):
        use_des_weights = self.temperature > 0.0
        weights = jax.lax.select(
            use_des_weights,
            get_temp_weights(self.popsize, self.temperature),
            get_recombination_weights(self.popsize),
        )
        return evox.State(
            key=key,
            sigma=self.sigma * jnp.ones(self.num_dims),
            center=self.center_init,
            weights=weights.reshape(-1, 1)
        )

    def ask(self, state):
        key, _ = jax.random.split(state.key)
        noise = jax.random.normal(key, (self.popsize, self.num_dims))
        x = state.center + noise * state.sigma.reshape(1, self.num_dims)
        return x, state.update(key=key,noise=noise, population=x)
    
    def tell(self, state, fitness):
        s = state.noise
        ranks = fitness.argsort()
        sorted_noise = s[ranks]
        grad_mean = (state.weights * sorted_noise).sum(axis=0)
        grad_sigma = (state.weights * (sorted_noise ** 2 - 1)).sum(axis=0)
        center = state.center + self.lrate_mean * state.sigma * grad_mean
        sigma = state.sigma * jnp.exp(self.lrate_sigma / 2 * grad_sigma)
        return state.update(center=center, sigma=sigma)



