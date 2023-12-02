# --------------------------------------------------------------------------------------
# 1. This code implements the algorithm described in the following paper:
# Title: Learn2Hop: Learned Optimization on Rough Landscapes
# Link: https://proceedings.mlr.press/v139/merchant21a.html
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
class ESMC(evox.Algorithm):
    def __init__(
        self,
        pop_size: int,
        center_init: jax.Array,
        optimizer: str = "adam",
        lrate_decay: float = 1.0,
        lrate_limit: float = 0.001,
        sigma_decay: float = 1.0,
        sigma_limit: float = 0.01,
        lr: float = 0.05,
        sigma: float = 0.03,
        init_min: float = 0.0,
        init_max: float = 0.0,
        clip_min: float = -jnp.finfo(jnp.float32).max,
        clip_max: float = jnp.finfo(jnp.float32).max,
    ):
        super().__init__()
        assert pop_size & 1
        self.num_dims = center_init.shape[0]
        self.center_init = center_init
        self.popsize = pop_size
        self.lr = lr
        self.sigma = sigma

        self.sigma_decay = sigma_decay
        self.sigma_limit = sigma_limit
        self.lrate_decay = lrate_decay
        self.lrate_limit = lrate_limit

        self.clip_max = init_max
        self.clip_min = init_min
        self.init_max = init_max
        self.init_min = init_min

        if optimizer == "adam":
            self.optimizer = optax.adam(learning_rate=lr)
        elif optimizer == "sgd":
            self.optimizer = optax.sgd(learning_rate=lr)
        else:
            raise NotImplementedError
        self.optimizer = evox.utils.OptaxWrapper(self.optimizer, center_init)

    def setup(self, key):
        return evox.State(
            key=key, center=self.center_init, sigma=jnp.ones(self.num_dims) * self.sigma
        )

    def ask(self, state):
        key, _ = jax.random.split(state.key)
        z_plus = jax.random.normal(
            state.key,
            (int(self.popsize / 2), self.num_dims),
        )
        z = jnp.concatenate([jnp.zeros((1, self.num_dims)), z_plus, -1.0 * z_plus])
        x = state.center + z * state.sigma.reshape(1, self.num_dims)
        return x, state.update(key=key, x=x)

    def tell(self, state, fitness):
        noise = (state.x - state.center) / state.sigma
        bline_fitness = fitness[0]
        noise = noise[1:]
        fitness = fitness[1:]
        noise_1 = noise[: int((self.popsize - 1) / 2)]
        fit_1 = fitness[: int((self.popsize - 1) / 2)]
        fit_2 = fitness[int((self.popsize - 1) / 2) :]
        fit_diff = jnp.minimum(fit_1, bline_fitness) - jnp.minimum(fit_2, bline_fitness)
        fit_diff_noise = jnp.dot(noise_1.T, fit_diff)
        theta_grad = 1.0 / int((self.popsize - 1) / 2) * fit_diff_noise

        updates, state = self.optimizer.update(state, theta_grad, state.center)
        center = optax.apply_updates(state.center, updates)

        sigma = state.sigma * self.sigma_decay
        sigma = jnp.maximum(sigma, self.sigma_limit)
        return state.update(center=center, sigma=sigma)
