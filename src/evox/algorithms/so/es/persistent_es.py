# --------------------------------------------------------------------------------------
# 1. This code implements the algorithm described in the following paper:
# Title: Unbiased Gradient Estimation in Unrolled Computation Graphs with Persistent Evolution Strategies
# Link: http://proceedings.mlr.press/v139/vicol21a.html
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
class PersistentES(evox.Algorithm):
    def __init__(
        self,
        pop_size: int,
        center_init: jax.Array,
        optimizer: str = "adam",
        lr: float = 0.05,
        sigma: float = 0.03,
        T: int = 100,
        K: int = 10,
        lrate_decay: float = 1.0,
        lrate_limit: float = 0.001,
        sigma_decay: float = 1.0,
        sigma_limit: float = 0.01,
        sigma_lrate: float = 0.2,
        sigma_max_change: float = 0.2,
        init_min: float = 0.0,
        init_max: float = 0.0,
        clip_min: float = -jnp.finfo(jnp.float32).max,
        clip_max: float = jnp.finfo(jnp.float32).max,
    ):
        super().__init__()
        assert pop_size % 2 == 0  # Population size must be even

        self.num_dims = center_init.shape[0]
        self.center_init = center_init
        self.popsize = pop_size
        self.lr = lr
        self.sigma = sigma
        self.T = T
        self.K = K

        self.sigma_decay = sigma_decay
        self.sigma_limit = sigma_limit
        self.lrate_decay = lrate_decay
        self.lrate_limit = lrate_limit
        self.clip_max = init_max
        self.clip_min = init_min
        self.init_max = init_max
        self.init_min = init_min
        self.sigma_max_change = sigma_max_change
        self.sigma_lrate = sigma_lrate

        if optimizer == "adam":
            self.optimizer = optax.adam(learning_rate=lr)
        elif optimizer == "sgd":
            self.optimizer = optax.sgd(learning_rate=lr)
        else:
            raise NotImplementedError

        self.optimizer = evox.utils.OptaxWrapper(self.optimizer, center_init)

    def setup(self, key):
        return evox.State(
            key=key,
            center=self.center_init,
            inner_step_counter=0,
            sigma=self.sigma,
            pert_accum=jnp.zeros((self.popsize, self.num_dims)),
        )

    def ask(self, state):
        key, _ = jax.random.split(state.key)
        pos_perts = (
            jax.random.normal(state.key, (self.popsize // 2, self.num_dims))
            * self.sigma
        )
        neg_perts = -pos_perts
        perts = jnp.concatenate([pos_perts, neg_perts], axis=0)
        # Add the perturbations from this unroll to the perturbation accumulators
        pert_accum = state.pert_accum + perts
        x = state.center + perts
        return x, state.update(key=key, pert_accum=pert_accum, population=x)

    def tell(self, state, fitness):
        theta_grad = jnp.mean(
            state.pert_accum * fitness.reshape(-1, 1) / (self.sigma**2),
            axis=0,
        )
        # Grad update using optimizer instance - decay lrate if desired
        updates, state = self.optimizer.update(state, theta_grad, state.center)
        center = optax.apply_updates(state.center, updates)

        inner_step_counter = state.inner_step_counter + self.K
        # Reset accumulated antithetic noise memory if done with inner problem
        reset = inner_step_counter >= self.T
        inner_step_counter = jax.lax.select(reset, 0, inner_step_counter)
        pert_accum = jax.lax.select(
            reset, jnp.zeros((self.popsize, self.num_dims)), state.pert_accum
        )

        sigma = self.sigma_decay * state.sigma
        sigma = jnp.maximum(sigma, self.sigma_limit)

        return state.update(
            center=center,
            sigma=sigma,
            pert_accum=pert_accum,
            inner_step_counter=inner_step_counter,
        )
