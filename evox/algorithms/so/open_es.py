import jax
import jax.numpy as jnp
from jax import lax

import evox as ex


@ex.jit_class
class OpenES(ex.Algorithm):
    def __init__(self, init_params, pop_size, learning_rate, noise_std):
        assert(noise_std > 0)
        assert(learning_rate > 0)
        assert(pop_size > 0)
        self.dim = init_params.shape[0]
        self.init_params = init_params
        self.pop_size = pop_size
        self.learning_rate = learning_rate
        self.noise_std = noise_std

    def _new_pop(self, key, params):
        key, noise_key = jax.random.split(key)
        noise = jax.random.normal(noise_key, shape=(self.pop_size, self.dim))
        population = jnp.repeat(params, self.pop_size).reshape(self.dim, self.pop_size).T
        population = population + self.noise_std * noise
        return population, noise, key

    def setup(self, key):
        population, noise, key = self._new_pop(key, self.init_params)
        return ex.State(population=population, params=self.init_params, noise=noise, key=key)

    def ask(self, state):
        return state, state.population

    def tell(self, state, fitness):
        params = state.params
        population, noise, key = self._new_pop(state.key, params)
        params = params + self.learning_rate / self.pop_size / self.noise_std * jnp.sum(jax.vmap(lambda F, e: F * e)(fitness, noise), axis=0)
        
        return state.update(population=population, params=params, key=key)
