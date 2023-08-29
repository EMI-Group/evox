# --------------------------------------------------------------------------------------
# 1. This code implements algorithms described in the following papers:
#
# Title: A new optimizer using particle swarm theory
# Link: https://ieeexplore.ieee.org/document/494215
# --------------------------------------------------------------------------------------

from functools import partial

import jax
import jax.numpy as jnp
import copy

from evox.utils import *
from evox import Algorithm, State, jit_class


@jit_class
class PSO(Algorithm):
    def __init__(
        self,
        lb,
        ub,
        pop_size,
        inertia_weight=0.6,
        cognitive_coefficient=2.5,
        social_coefficient=0.8,
        mean=None,
        stdev=None,
    ):
        self.dim = lb.shape[0]
        self.lb = lb
        self.ub = ub
        self.pop_size = pop_size
        self.w = inertia_weight
        self.phi_p = cognitive_coefficient
        self.phi_g = social_coefficient
        self.mean = mean
        self.stdev = stdev

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
        population = jnp.clip(population, self.lb, self.ub)

        return state.update(
            population=population,
            velocity=velocity,
            local_best_location=local_best_location,
            local_best_fitness=local_best_fitness,
            global_best_location=global_best_location,
            global_best_fitness=global_best_fitness,
            key=key,
        )
