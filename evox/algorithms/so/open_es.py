import jax
import jax.numpy as jnp
from jax import lax
import evox as ex


@ex.jit_class
class OpenES(ex.Algorithm):
    def __init__(
        self, init_params, pop_size, learning_rate, noise_std, mirrored_sampling=True
    ):
        """
        Implement the algorithm described in "Evolution Strategies as a Scalable Alternative to Reinforcement Learning"
        from https://arxiv.org/abs/1703.03864
        """
        assert noise_std > 0
        assert learning_rate > 0
        assert pop_size > 0

        if mirrored_sampling is True:
            assert (
                pop_size % 2 == 0
            ), "When mirrored_sampling is True, pop_size must be a multiple of 2."

        self.dim = init_params.shape[0]
        self.init_params = init_params
        self.pop_size = pop_size
        self.learning_rate = learning_rate
        self.noise_std = noise_std
        self.mirrored_sampling = mirrored_sampling

    def setup(self, key):
        # placeholder
        population = jnp.tile(self.init_params, (self.pop_size, 1))
        noise = jnp.tile(self.init_params, (self.pop_size, 1))
        return ex.State(
            population=population, params=self.init_params, noise=noise, key=key
        )

    def ask(self, state):
        key, noise_key = jax.random.split(state.key)
        if self.mirrored_sampling:
            noise = jax.random.normal(noise_key, shape=(self.pop_size // 2, self.dim))
            noise = jnp.concatenate([noise, -noise], axis=0)
        else:
            noise = jax.random.normal(noise_key, shape=(self.pop_size, self.dim))
        population = state.params[jnp.newaxis, :] + self.noise_std * noise

        return (
            state.update(population=population, key=key, noise=noise),
            population,
        )

    def tell(self, state, fitness):
        params = state.params - self.learning_rate / self.noise_std * jnp.mean(
            fitness[:, jnp.newaxis] * state.noise, axis=0
        )
        return state.update(params=params)
