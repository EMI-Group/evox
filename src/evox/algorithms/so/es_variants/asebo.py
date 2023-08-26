# --------------------------------------------------------------------------------------
# 1. This code implements the algorithm described in the following paper:
#
# Title: From Complexity to Simplicity: Adaptive ES-Active Subspaces for Blackbox Optimization
# Link: https://arxiv.org/abs/1903.04268
#
# 2. This code has been inspired by or utilizes the algorithmic implementation from evosax.
#
# More information about evosax can be found at the following URL:
# GitHub Link: https://github.com/RobertTLange/evosax
# --------------------------------------------------------------------------------------

from typing import Tuple, Optional, Union
import jax
import jax.numpy as jnp
import evox
import optax


class ASEBO(evox.Algorithm):
    def __init__(
        self,
        pop_size: int,
        center_init: jax.Array,
        optimizer: str = "adam",
        lr: float = 0.05,
        lr_decay: float = 1.0,
        lr_limit: float = 0.001,
        sigma: float = 0.03,
        sigma_decay: float = 1.0,
        sigma_limit: float = 0.01,
        mean_decay: float = 0.0,
        subspace_dims: int = 50,
    ):

        super().__init__()
        assert not pop_size & 1

        if optimizer == "adam":
            self.optimizer = optax.adam(learning_rate=lr)
        else:
            raise NotImplementedError
        self.optimizer = evox.utils.OptaxWrapper(self.optimizer, center_init)
        self.dim = center_init.shape[0]
        self.center_init = center_init
        self.pop_size = pop_size
        self.lr = lr
        self.sigma = sigma
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_limit = lr_limit
        self.sigma = sigma
        self.sigma_decay = sigma_decay
        self.sigma_limit = sigma_limit
        self.mean_decay = mean_decay
        self.subspace_dims = subspace_dims

    def setup(self, key):
        grad_subspace = jnp.zeros((self.subspace_dims, self.dim))
        return evox.State(
            key=key,
            center=self.center_init,
            grad_subspace=grad_subspace,
            gen_counter=0,
            sigma=self.sigma,
            alpha=0.1,
            x=self.center_init,
            UUT=jnp.zeros((self.dim, self.dim)),
            UUT_ort=jnp.zeros((self.dim, self.dim)),
        )

    def ask(self, state):
        key, rng = jax.random.split(state.key)
        X = state.grad_subspace
        X -= jnp.mean(X, axis=0)
        U, S, Vt = jnp.linalg.svd(X, full_matrices=False)

        def svd_flip(u, v):
            max_abs_cols = jnp.argmax(jnp.abs(u), axis=0)
            signs = jnp.sign(u[max_abs_cols, jnp.arange(u.shape[1])])
            u *= signs
            v *= signs[:, jnp.newaxis]
            return u, v

        U, Vt = svd_flip(U, Vt)
        U = Vt[: int(self.pop_size / 2)]
        UUT = jnp.matmul(U.T, U)
        U_ort = Vt[int(self.pop_size / 2) :]
        UUT_ort = jnp.matmul(U_ort.T, U_ort)
        subspace_ready = state.gen_counter > self.subspace_dims

        UUT = jax.lax.select(subspace_ready, UUT, jnp.zeros((self.dim, self.dim)))
        cov = (
            state.sigma * (state.alpha / self.dim) * jnp.eye(self.dim)
            + ((1 - state.alpha) / int(self.pop_size / 2)) * UUT
        )
        chol = jnp.linalg.cholesky(cov)
        noise = jax.random.normal(rng, (self.dim, int(self.pop_size / 2)))
        z_plus = jnp.swapaxes(chol @ noise, 0, 1)
        z_plus /= jnp.linalg.norm(z_plus, axis=-1)[:, jnp.newaxis]
        z = jnp.concatenate([z_plus, -1.0 * z_plus])
        x = state.center + z
        return (
            x,
            state.update(
                key=key,
                population=x,
                noise=z,
                UUT=UUT,
                UUT_ort=UUT_ort,
                gen_counter=state.gen_counter + 1,
            ),
        )

    def tell(self, state, fitness):
        noise = (state.population - state.center) / state.sigma
        noise_1 = noise[: int(self.pop_size / 2)]
        fit_1 = fitness[: int(self.pop_size / 2)]
        fit_2 = fitness[int(self.pop_size / 2) :]
        fit_diff_noise = jnp.dot(noise_1.T, fit_1 - fit_2)
        theta_grad = 1.0 / 2.0 * fit_diff_noise
        alpha = jnp.linalg.norm(jnp.dot(theta_grad, state.UUT_ort)) / jnp.linalg.norm(
            jnp.dot(theta_grad, state.UUT)
        )
        subspace_ready = state.gen_counter > self.subspace_dims
        alpha = jax.lax.select(subspace_ready, alpha, 1.0)
        grad_subspace = jnp.zeros((self.subspace_dims, self.dim))
        grad_subspace = grad_subspace.at[:-1, :].set(state.grad_subspace[1:, :])
        grad_subspace = grad_subspace.at[-1, :].set(theta_grad)
        theta_grad /= jnp.linalg.norm(theta_grad) / self.dim + 1e-8
        updates, state = self.optimizer.update(state, theta_grad, state.center)
        center = optax.apply_updates(state.center, updates)
        sigma = state.sigma * self.sigma_decay
        sigma = jnp.maximum(sigma, self.sigma_limit)
        return state.update(center=center, sigma=sigma, alpha=alpha)
