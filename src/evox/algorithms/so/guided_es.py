# --------------------------------------------------------------------------------------
# 1. This code implements the algorithm described in the following paper:
# Title: Guided evolutionary strategies: Augmenting random search with surrogate gradients
# Link: https://arxiv.org/abs/1806.10230
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
class GuidedES(evox.Algorithm):
    def __init__(
        self,
        pop_size: int,
        center_init,
        subspace_dims: int = 1,
        optimizer: str = "sgd",
        sigma_init: float = 0.03,
        lrate_init: float = 60,
        lrate_decay: float = 1.0,
        lrate_limit: float = 0.001,
        sigma_decay: float = 1.0,
        sigma_limit: float = 0.01,
        mean_decay: float = 0.0,
    ):
        super().__init__()
        assert pop_size % 2 == 0
        self.num_dims = center_init.shape[0]
        self.center_init = center_init
        self.popsize = pop_size
        self.lr = lrate_init
        self.sigma = sigma_init
        self.subspace_dims = subspace_dims
        self.sigma_decay = sigma_decay
        self.sigma_limit = sigma_limit
        self.lrate_decay = lrate_decay
        self.lrate_limit = lrate_limit
        self.alpha = 0.5
        self.beta = 1.0
        if optimizer == "adam":
            self.optimizer = optax.adam(learning_rate=lrate_init)
        elif optimizer == "sgd":
            self.optimizer = optax.sgd(learning_rate=lrate_init)
        else:
            raise NotImplementedError
        self.optimizer = evox.utils.OptaxWrapper(self.optimizer, center_init)

    def setup(self, key):
        grad_subspace = jax.random.normal(key, (self.subspace_dims, self.num_dims))
        # opt_state=evox.State()
        return evox.State(
            key=key,
            center=self.center_init,
            sigma=self.sigma,
            grad_subspace=grad_subspace,
        )

    def ask(self, state):
        key, _ = jax.random.split(state.key)
        a = state.sigma * jnp.sqrt(self.alpha / self.num_dims)
        c = state.sigma * jnp.sqrt((1.0 - self.alpha) / self.subspace_dims)
        key_full, key_sub = jax.random.split(state.key, 2)
        eps_full = jax.random.normal(
            key_full, shape=(self.num_dims, int(self.popsize / 2))
        )
        eps_subspace = jax.random.normal(
            key_sub, shape=(self.subspace_dims, int(self.popsize / 2))
        )
        Q, _ = jnp.linalg.qr(state.grad_subspace)
        # Antithetic sampling of noise
        z_plus = a * eps_full + c * jnp.dot(Q, eps_subspace)
        z_plus = jnp.swapaxes(z_plus, 0, 1)
        z = jnp.concatenate([z_plus, -1.0 * z_plus])
        x = state.center + z
        return x, state.update(key=key, z=z)

    def tell(self, state, fitness):
        noise = state.z / state.sigma
        noise_1 = noise[: int(self.popsize / 2)]
        fit_1 = fitness[: int(self.popsize / 2)]
        fit_2 = fitness[int(self.popsize / 2) :]
        fit_diff = fit_1 - fit_2
        fit_diff_noise = jnp.dot(noise_1.T, fit_diff)
        theta_grad = (self.beta / self.popsize) * fit_diff_noise

        grad_subspace = jnp.zeros((self.subspace_dims, self.num_dims))
        grad_subspace = grad_subspace.at[:-1, :].set(state.grad_subspace[1:, :])
        grad_subspace = grad_subspace.at[-1, :].set(theta_grad)
        state = state.update(grad_subspace=grad_subspace)

        # Grad update using optimizer instance - decay lrate if desired
        updates, state = self.optimizer.update(state, theta_grad, state.center)
        center = optax.apply_updates(state.center, updates)

        # lrate=self.lrate_decay*state.opt_state.lrate
        # lrate= jnp.maximum(lrate, self.lrate_limit)
        # Update lrate and standard deviation based on min and decay
        sigma = self.sigma_decay * state.sigma
        sigma = jnp.maximum(sigma, self.sigma_limit)

        state = state.update(center=center, sigma=sigma)
        return state
