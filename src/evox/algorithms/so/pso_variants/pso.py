# --------------------------------------------------------------------------------------
# 1. This code implements algorithms described in the following papers:
#
# Title: A new optimizer using particle swarm theory
# Link: https://ieeexplore.ieee.org/document/494215
# --------------------------------------------------------------------------------------

from typing import Optional

import jax
import jax.numpy as jnp

from evox import Algorithm, State, dataclass, pytree_field
from evox.utils import *


@dataclass
class PSO(Algorithm):
    dim: jax.Array = pytree_field(static=True, init=False)
    lb: jax.Array
    ub: jax.Array
    pop_size: jax.Array = pytree_field(static=True)
    w: jax.Array = pytree_field(default=0.6)
    phi_p: jax.Array = pytree_field(default=2.5)
    phi_g: jax.Array = pytree_field(default=0.8)
    mean: Optional[jax.Array] = pytree_field(default=None)
    stdev: Optional[jax.Array] = pytree_field(default=None)
    bound_method: str = pytree_field(static=True, default="clip")

    def __post_init__(self):
        self.set_frozen_attr("dim", self.lb.shape[0])

    def setup(self, key):
        state_key, init_pop_key, init_v_key = jax.random.split(key, 3)
        if self.mean is not None and self.stdev is not None:
            population = self.stdev * jax.random.normal(
                init_pop_key, shape=(self.pop_size, self.dim)
            )
            population = jnp.clip(population, self.lb, self.ub)
            velocity = self.stdev * jax.random.normal(
                init_v_key, shape=(self.pop_size, self.dim)
            )
        else:
            length = self.ub - self.lb
            population = jax.random.uniform(
                init_pop_key, shape=(self.pop_size, self.dim)
            )
            population = population * length + self.lb
            velocity = jax.random.uniform(init_v_key, shape=(self.pop_size, self.dim))
            velocity = velocity * length * 2 - length

        return State(
            population=population,
            velocity=velocity,
            local_best_location=population,
            local_best_fitness=jnp.full((self.pop_size,), jnp.inf),
            global_best_location=population[0],
            global_best_fitness=jnp.array([jnp.inf]),
            key=state_key,
        )

    def ask(self, state):
        return state.population, state

    def tell(self, state, fitness):
        key, rg_key, rp_key = jax.random.split(state.key, 3)

        rg = jax.random.uniform(rg_key, shape=(self.pop_size, self.dim))
        rp = jax.random.uniform(rp_key, shape=(self.pop_size, self.dim))

        compare = state.local_best_fitness > fitness
        local_best_location = jnp.where(
            compare[:, jnp.newaxis], state.population, state.local_best_location
        )
        local_best_fitness = jnp.minimum(state.local_best_fitness, fitness)

        global_best_location, global_best_fitness = min_by(
            [state.global_best_location[jnp.newaxis, :], state.population],
            [state.global_best_fitness, fitness],
        )

        global_best_fitness = jnp.atleast_1d(global_best_fitness)

        velocity = (
            self.w * state.velocity
            + self.phi_p * rp * (local_best_location - state.population)
            + self.phi_g * rg * (global_best_location - state.population)
        )
        population = state.population + velocity

        if self.bound_method == "clip":
            population = jnp.clip(population, self.lb, self.ub)
        elif self.bound_method == "reflect":
            lower_bound_violation = population < self.lb
            upper_bound_violation = population > self.ub

            population = jnp.where(
                lower_bound_violation, 2 * self.lb - population, population
            )
            population = jnp.where(
                upper_bound_violation, 2 * self.ub - population, population
            )

            velocity = jnp.where(
                lower_bound_violation | upper_bound_violation, -velocity, velocity
            )

        return state.replace(
            population=population,
            velocity=velocity,
            local_best_location=local_best_location,
            local_best_fitness=local_best_fitness,
            global_best_location=global_best_location,
            global_best_fitness=global_best_fitness,
            key=key,
        )
