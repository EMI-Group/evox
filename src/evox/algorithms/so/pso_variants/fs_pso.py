# --------------------------------------------------------------------------------------
# 1. This code implements algorithms described in the following papers:
#
# Title: Feature Selection Based on Hybridization of Genetic Algorithm and Particle Swarm Optimization
# Link: https://ieeexplore.ieee.org/document/6866865
# --------------------------------------------------------------------------------------

from functools import partial

import jax
import jax.numpy as jnp
import copy

from evox.utils import *
from evox import Algorithm, State, jit_class


# FS-PSO: Feature Selection PSO
@jit_class
class FSPSO(Algorithm):
    def __init__(
        self,
        lb,  # lower bound of problem
        ub,  # upper bound of problem
        pop_size,  # population size
        inertia_weight=0.6,  # w
        cognitive_coefficient=2.5,  # c
        social_coefficient=0.8,  # s
        mean=None,
        stdev=None,
        mutate_rate=0.01,  # mutation ratio
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
        self.mutate_rate = mutate_rate

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
        key, rg_key, rp_key, tn_key, mu_key, ma_key = jax.random.split(state.key, 6)

        # --------------Enhancement------------
        ranked_index = jnp.argsort(fitness)
        elite_index = ranked_index[: self.pop_size // 2]
        ranked_population = state.population[ranked_index]
        ranked_velocity = state.velocity[ranked_index]
        elite_population = state.population[elite_index]
        elite_velocity = state.velocity[elite_index]
        elite_fitness = fitness[elite_index]
        elite_lbest_location = state.local_best_location[elite_index]
        elite_lbest_fitness = state.local_best_fitness[elite_index]
        rg = jax.random.uniform(rg_key, shape=(self.pop_size // 2, self.dim))
        rp = jax.random.uniform(rp_key, shape=(self.pop_size // 2, self.dim))

        compare = elite_lbest_fitness > elite_fitness
        lbest_location = jnp.where(
            compare[:, jnp.newaxis], elite_population, elite_lbest_location
        )
        lbest_fitness = jnp.minimum(elite_lbest_fitness, elite_fitness)

        global_best_location, global_best_fitness = min_by(
            [state.global_best_location[jnp.newaxis, :], elite_population],
            [state.global_best_fitness, elite_fitness],
        )

        global_best_fitness = jnp.atleast_1d(global_best_fitness)

        updated_elite_velocity = (
            self.w * elite_velocity
            + self.phi_p * rp * (elite_lbest_location - elite_population)
            + self.phi_g * rg * (global_best_location - elite_population)
        )
        updated_elite_population = elite_population + updated_elite_velocity
        updated_elite_population = jnp.clip(updated_elite_population, self.lb, self.ub)
        # ----------------Crossover----------------
        tournament1 = jax.random.choice(
            tn_key,
            jnp.arange(0, elite_index.shape[0]),
            (1, self.pop_size - (self.pop_size // 2)),
        )
        tournament2 = jax.random.choice(
            tn_key,
            jnp.arange(0, elite_index.shape[0]),
            (1, self.pop_size - (self.pop_size // 2)),
        )
        compare = elite_fitness[tournament1] < elite_fitness[tournament2]
        mutating_pool = jnp.where(compare, tournament1, tournament2)
        # -Extend (mutate and create new generation)-
        unmutated_population = elite_population[mutating_pool.flatten()]
        offspring_velocity = elite_velocity[mutating_pool.flatten()]

        offset = jax.random.uniform(
            key=mu_key,
            shape=(unmutated_population.shape[0], self.dim),
            minval=-1,
            maxval=1,
        ) * (self.ub - self.lb)
        mp = jax.random.uniform(ma_key, (unmutated_population.shape[0], self.dim))
        mask = mp < self.mutate_rate
        offspring_population = unmutated_population + jnp.where(
            mask, offset, jnp.zeros((unmutated_population.shape[0], self.dim))
        )
        offspring_population = jnp.clip(offspring_population, self.lb, self.ub)
        offspring_lbest_location = offspring_population
        offspring_lbest_fitness = jnp.full((offspring_population.shape[0],), jnp.inf)

        new_population = jnp.concatenate(
            (updated_elite_population, offspring_population)
        )
        new_velocity = jnp.concatenate((updated_elite_velocity, offspring_velocity))
        new_lbest_location = jnp.concatenate((lbest_location, offspring_lbest_location))
        new_lbest_fitness = jnp.concatenate((lbest_fitness, offspring_lbest_fitness))

        return state.update(
            population=new_population,
            velocity=new_velocity,
            local_best_location=new_lbest_location,
            local_best_fitness=new_lbest_fitness,
            global_best_location=global_best_location,
            global_best_fitness=global_best_fitness,
            key=key,
        )
