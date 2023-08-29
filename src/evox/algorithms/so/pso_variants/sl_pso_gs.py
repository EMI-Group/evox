# --------------------------------------------------------------------------------------
# 1. This code implements algorithms described in the following papers:
#
# Title: Demonstrator selection in a social learning particle swarm optimizer
# Link: https://ieeexplore.ieee.org/document/6900227
# --------------------------------------------------------------------------------------

import jax
import jax.numpy as jnp

from evox.utils import *
from evox import Algorithm, State, jit_class

# SL-PSO: Social Learning PSO
# SL-PSO-GS: Using Gaussian Sampling for Demonstator Choice
@jit_class
class SLPSOGS(Algorithm):
    def __init__(
        self,
        lb, # lower bound of problem
        ub, # upper bound of problem
        pop_size,
        social_influence_factor, # epsilon
        demonstrator_choice_factor, # theta
    ):
        self.dim = lb.shape[0]
        self.lb = lb
        self.ub = ub
        self.pop_size = pop_size
        self.social_influence_factor = social_influence_factor
        self.demonstrator_choice_factor = demonstrator_choice_factor

    def setup(self, key):
        state_key, init_pop_key, init_v_key = jax.random.split(key, 3)
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
            global_best_location=population[0],
            global_best_fitness=jnp.array([jnp.inf]),
            key=state_key,
        )

    def ask(self, state):
        return state.population, state

    def tell(self, state, fitness):
        key, r1_key, r2_key, r3_key, demonstrator_choice_key = jax.random.split(state.key, num=5)

        r1 = jax.random.uniform(r1_key, shape=(self.pop_size, self.dim))
        r2 = jax.random.uniform(r2_key, shape=(self.pop_size, self.dim))
        r3 = jax.random.uniform(r3_key, shape=(self.pop_size, self.dim))

        global_best_location, global_best_fitness = min_by(
            [state.global_best_location[jnp.newaxis, :], state.population],
            [state.global_best_fitness, fitness],
        )
        global_best_fitness = jnp.atleast_1d(global_best_fitness)

        # ----------------- Demonstator Choice -----------------
        # sort from largest fitness to smallest fitness (worst to best)
        ranked_population = state.population[jnp.argsort(-fitness)]
        sigma = self.demonstrator_choice_factor * (self.pop_size - (jnp.arange(self.pop_size) + 1))
        standard_normal_distribution = jax.random.normal(demonstrator_choice_key, shape=(self.pop_size,))
        # normal distribution (shape=(self.pop_size,)) means
        # each individual choose a demonstrator by normal distribution
        # with mean = pop_size and std = sigma
        normal_distribution = sigma * (-jnp.abs(standard_normal_distribution)) + self.pop_size
        index_k = jnp.floor(jnp.clip(normal_distribution, 1, self.pop_size)).astype(int) - 1
        X_k = ranked_population[index_k]
        # ------------------------------------------------------

        X_avg = jnp.mean(state.population, axis=0)
        velocity = (
            r1 * state.velocity
            + r2 * (X_k - state.population)
            + r3 * self.social_influence_factor * (X_avg - state.population)
        )
        population = state.population + velocity
        population = jnp.clip(population, self.lb, self.ub)
        return state.update(
            population=population,
            velocity=velocity,
            global_best_location=global_best_location,
            global_best_fitness=global_best_fitness,
            key=key,
        )
