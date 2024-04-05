# --------------------------------------------------------------------------------------
# 1. This code implements the algorithm described in the following paper:
# Title: Parameter-exploring Policy Gradients (PGPE)
# Link: https://mediatum.ub.tum.de/doc/1287490/file.pdf
# --------------------------------------------------------------------------------------

from dataclasses import field
from typing import Union

import jax
import jax.numpy as jnp
import optax
from jax import jit, lax
from jax.tree_util import tree_map, tree_reduce

from evox import (Algorithm, State, Stateful, Static, dataclass, jit_class,
                  use_state, utils)


@jit
def tree_l2_norm(pytree):
    return jnp.sqrt(
        tree_reduce(lambda x, y: x + y, tree_map(lambda leaf: jnp.sum(leaf**2), pytree))
    )


@jit_class
@dataclass
class ClipUp(Stateful):
    step_size: float
    max_speed: float
    momentum: float
    params: jax.Array

    def setup(self, key):
        velocity = tree_map(lambda x: jnp.zeros_like(x), self.params)
        return State(velocity=velocity)

    def update(self, state, gradient, _params=None):
        grad_norm = tree_l2_norm(gradient)
        velocity = tree_map(
            lambda v, g: self.momentum * v + self.step_size * g / grad_norm,
            state.velocity,
            gradient,
        )
        velocity_norm = tree_l2_norm(velocity)

        def clip_velocity(velocity):
            return tree_map(lambda v: self.max_speed * v / velocity_norm, velocity)

        velocity = lax.cond(
            velocity_norm > self.max_speed,
            clip_velocity,
            lambda v: v,  # identity function
            velocity,
        )
        return -velocity, state.update(velocity=velocity)


@jit_class
@dataclass
class PGPE(Algorithm):
    pop_size: Static[int]
    center_init: jax.Array
    optimizer: Union[str, optax.GradientTransformation, Stateful]
    stdev_init: float = 0.1
    center_learning_rate: float = 0.15
    stdev_learning_rate: float = 0.1
    stdev_max_change: float = 0.2
    dim: Static[int] = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "dim", self.center_init.shape[0])
        if isinstance(self.optimizer, str):
            if self.optimizer == "clipup":
                optimizer = ClipUp(
                    step_size=0.15, max_speed=0.3, momentum=0.9, params=self.center_init
                )
            elif hasattr(optax, self.optimizer):
                optimizer = utils.OptaxWrapper(
                    getattr(optax, self.optimizer)(
                        learning_rate=self.center_learning_rate
                    ),
                    self.center_init,
                )
            else:
                raise ValueError(f"Unknown optimizer {self.optimizer}")
            object.__setattr__(self, "optimizer", optimizer)
        elif isinstance(self.optimizer, optax.GradientTransformation):
            object.__setattr__(
                self, "optimizer", utils.OptaxWrapper(self.optimizer, self.center_init)
            )
        elif not isinstance(self.optimizer, Stateful):
            raise TypeError(f"{self.optimizer} is not supported right now")
        assert isinstance(self.optimizer, Stateful)

    def setup(self, key):
        return State(
            center=self.center_init,
            stdev=jnp.full((self.dim,), self.stdev_init),
            key=key,
            noise=jnp.empty((self.pop_size // 2, self.dim)),
        )

    def ask(self, state):
        key, subkey = jax.random.split(state.key)
        noise = jax.random.normal(subkey, (self.pop_size // 2, self.dim)) * state.stdev
        D = jnp.concatenate([state.center + noise, state.center - noise], axis=0)
        return D, state.update(key=key, noise=noise)

    def tell(self, state, fitness):
        F_pos = fitness[: self.pop_size // 2]
        F_neg = fitness[self.pop_size // 2 :]

        delta_x = jnp.mean(((F_pos - F_neg) / 2)[:, jnp.newaxis] * state.noise, axis=0)
        f_avg = jnp.mean(fitness)
        delta_stdev = jnp.mean(
            ((F_pos + F_neg) / 2 - f_avg)[:, jnp.newaxis]
            * ((state.noise**2 - state.stdev**2) / state.stdev),
            axis=0,
        )
        updates, state = use_state(self.optimizer.update)(state, delta_x, state.center)
        center = optax.apply_updates(state.center, updates)
        stdev_updates = self.stdev_learning_rate * delta_stdev
        bound = jnp.abs(state.stdev * self.stdev_max_change)
        stdev_updates = jnp.clip(stdev_updates, -bound, bound)
        return state.update(
            center=center,
            stdev=state.stdev - stdev_updates,
        )
