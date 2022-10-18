from functools import partial

import jax
import jax.numpy as jnp
import copy
import chex
import optax

import evoxlib as exl


@exl.jit_class
class PGPE(exl.Algorithm):
    def __init__(
        self,
        pop_size: int,
        center_init: jnp.ndarray,
        optimizer: str,
        stdev_init: float = 0.1,
        center_learning_rate: float = 0.15,
        stdev_learning_rate: float = 0.1,
        stdev_max_change: float = 0.2,
    ):
        self.dim = center_init.shape[0]
        self.pop_size = pop_size
        self.center_init = center_init
        self.stdev = jnp.full_like(center_init, stdev_init)
        self.stdev_learning_rate = stdev_learning_rate
        self.stdev_max_change = stdev_max_change
        if optimizer == "adam":
            self.optimizer = optax.adam(learning_rate=center_learning_rate)
        else:
            raise TypeError(f"{optimizer} is not supported right now")

    def setup(self, key):
        opt_state = self.optimizer.init(self.center_init)
        return exl.State(
            center=self.center_init,
            stdev=self.stdev,
            opt_state=opt_state,
            key=key,
            sample=None,
        )

    def ask(self, state):
        key, subkey = jax.random.split(state.key)
        delta = (
            jax.random.normal(state.key, (self.pop_size // 2, self.dim)) * state.stdev
        )
        D = jnp.concatenate([state.center + delta, state.center - delta], axis=0)
        return state.update(key=subkey, sample=D), D

    def tell(self, state, fitness):
        D_pos = state.sample[: self.pop_size // 2, :]
        F_pos = fitness[: self.pop_size // 2]
        F_neg = fitness[self.pop_size // 2 :]

        delta_x = jnp.mean(
            ((F_pos - F_neg) / 2)[:, jnp.newaxis] * (D_pos - state.center), axis=0
        )
        f_avg = jnp.mean(fitness)
        delta_stdev = jnp.mean(
            ((F_pos + F_neg) / 2 - f_avg)[:, jnp.newaxis]
            * (((D_pos - state.center) ** 2 - state.stdev**2) / state.stdev),
            axis=0,
        )
        updates, opt_state = self.optimizer.update(
            delta_x, state.opt_state, state.center
        )
        center = optax.apply_updates(state.center, updates)
        stdev_updates = self.stdev_learning_rate * delta_stdev
        bound = jnp.abs(state.stdev * self.stdev_max_change)
        stdev_updates = jnp.clip(stdev_updates, -bound, bound)
        return state.update(
            center=center,
            stdev=state.stdev - stdev_updates,
            opt_state=opt_state,
        )
