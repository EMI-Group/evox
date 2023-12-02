# --------------------------------------------------------------------------------------
# This code implements algorithms described in the following papers:
#
# Title: Exponential Natural Evolution Strategies (XNES)
# Link: https://dl.acm.org/doi/abs/10.1145/1830483.1830557
#
# Title: Natural Evolution Strategies (SeparableNES)
# Link: https://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf
# --------------------------------------------------------------------------------------

import jax
import jax.numpy as jnp
import optax

import evox


@evox.jit_class
class OpenES(evox.Algorithm):
    def __init__(
        self,
        center_init,
        pop_size,
        learning_rate,
        noise_stdev,
        optimizer=None,
        mirrored_sampling=True,
    ):
        """
        Implement the algorithm described in "Evolution Strategies as a Scalable Alternative to Reinforcement Learning"
        from https://arxiv.org/abs/1703.03864
        """
        assert noise_stdev > 0
        assert learning_rate > 0
        assert pop_size > 0

        if mirrored_sampling is True:
            assert (
                pop_size % 2 == 0
            ), "When mirrored_sampling is True, pop_size must be a multiple of 2."

        self.dim = center_init.shape[0]
        self.center_init = center_init
        self.pop_size = pop_size
        self.learning_rate = learning_rate
        self.noise_stdev = noise_stdev
        self.mirrored_sampling = mirrored_sampling

        if optimizer == "adam":
            self.optimizer = evox.utils.OptaxWrapper(
                optax.adam(learning_rate=learning_rate), center_init
            )
        else:
            self.optimizer = None

    def setup(self, key):
        # placeholder
        population = jnp.tile(self.center_init, (self.pop_size, 1))
        noise = jnp.tile(self.center_init, (self.pop_size, 1))
        return evox.State(
            population=population, center=self.center_init, noise=noise, key=key
        )

    def ask(self, state):
        key, noise_key = jax.random.split(state.key)
        if self.mirrored_sampling:
            noise = jax.random.normal(noise_key, shape=(self.pop_size // 2, self.dim))
            noise = jnp.concatenate([noise, -noise], axis=0)
        else:
            noise = jax.random.normal(noise_key, shape=(self.pop_size, self.dim))
        population = state.center[jnp.newaxis, :] + self.noise_stdev * noise

        return population, state.update(population=population, key=key, noise=noise)

    def tell(self, state, fitness):
        grad = state.noise.T @ fitness / self.pop_size / self.noise_stdev
        if self.optimizer is None:
            center = state.center - self.learning_rate * grad
        else:
            updates, state = self.optimizer.update(state, state.center)
            center = optax.apply_updates(state.center, updates)
        return state.update(center=center)
