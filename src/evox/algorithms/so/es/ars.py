# --------------------------------------------------------------------------------------
# 1. This code implements algorithms described in the following papers:
#
# Title: Simple random search provides a competitive approach to reinforcement learning(ARS)
# Link: https://arxiv.org/pdf/1803.07055.pdf
#
# 2. This code has been inspired by or utilizes the algorithmic implementation from evosax.
# More information about evosax can be found at the following URL:
# GitHub Link: https://github.com/RobertTLange/evosax
# --------------------------------------------------------------------------------------

import jax
import jax.numpy as jnp
import optax
import evox


@evox.jit_class
class ARS(evox.Algorithm):
    def __init__(
        self,
        pop_size: int,
        center_init: jax.Array,
        elite_ratio: float = 0.1,
        optimizer: str = "adam",
        lr: float = 0.05,
        sigma: float = 0.03,
    ):
        super().__init__()

        assert not pop_size & 1
        assert 0 <= elite_ratio <= 1

        self.dim = center_init.shape[0]
        self.center_init = center_init
        self.pop_size = pop_size
        self.lr = lr
        self.sigma = sigma

        self.elite_ratio = elite_ratio
        self.elite_pop_size = max(1, int(self.pop_size / 2 * self.elite_ratio))

        if optimizer == "adam":
            self.optimizer = optax.adam(learning_rate=lr)
        else:
            raise NotImplementedError

        self.optimizer = evox.utils.OptaxWrapper(self.optimizer, center_init)

    def setup(self, key):
        return evox.State(
            key=key,
            center=self.center_init,
        )

    def ask(self, state):
        key, _ = jax.random.split(state.key)
        z_plus = jax.random.normal(
            state.key,
            (int(self.pop_size / 2), self.dim),
        )
        z = jnp.concatenate([z_plus, -1.0 * z_plus])
        x = state.center + self.sigma * z
        return x, state.update(key=key, population=x, noise=z)

    def tell(self, state, fitness):
        noise_1 = state.noise[: int(self.pop_size / 2)]
        fit_1 = fitness[: int(self.pop_size / 2)]
        fit_2 = fitness[int(self.pop_size / 2) :]
        elite_idx = jnp.minimum(fit_1, fit_2).argsort()[: self.elite_pop_size]

        fitness_elite = jnp.concatenate([fit_1[elite_idx], fit_2[elite_idx]])
        # Add small constant to ensure non-zero division stability
        sigma_fitness = jnp.std(fitness_elite) + 1e-05
        fit_diff = fit_1[elite_idx] - fit_2[elite_idx]
        fit_diff_noise = jnp.dot(noise_1[elite_idx].T, fit_diff)
        theta_grad = 1.0 / (self.elite_pop_size * sigma_fitness) * fit_diff_noise

        updates, state = self.optimizer.update(state, theta_grad, state.center)
        center = optax.apply_updates(state.center, updates)

        return state.update(center=center)
