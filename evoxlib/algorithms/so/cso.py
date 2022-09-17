from functools import partial

import jax
import jax.numpy as jnp
import copy

import evoxlib as exl


@exl.jit_class
class CSO(exl.Algorithm):
    def __init__(self, lb, ub, pop_size, phi=0.1):
        self.dim = lb.shape[0]
        self.lb = lb
        self.ub = ub
        self.pop_size = pop_size
        self.phi = phi

    def setup(self, key):
        state_key, init_key = jax.random.split(key)
        population = jax.random.uniform(init_key, shape=(self.pop_size, self.dim))
        population = population * (self.ub - self.lb) + self.lb
        speed = jnp.zeros(shape=(self.pop_size, self.dim))

        return exl.State(population=population, speed=speed, key=state_key)

    def ask(self, state):
        return state, state.population

    def tell(self, state, fitness):
        key, pairing_key, lambda1_key, lambda2_key, lambda3_key = jax.random.split(
            state.key, num=5
        )
        randperm = jax.random.permutation(pairing_key, self.pop_size).reshape(2, -1)
        mask = fitness[randperm[0, :]] < fitness[randperm[1, :]]

        teachers = jnp.where(mask, randperm[0, :], randperm[1, :])
        students = jnp.where(mask, randperm[1, :], randperm[0, :])

        center = jnp.mean(state.population, axis=0)

        lambda1 = jax.random.uniform(lambda1_key, shape=(self.pop_size // 2, self.dim))
        lambda2 = jax.random.uniform(lambda2_key, shape=(self.pop_size // 2, self.dim))
        lambda3 = jax.random.uniform(lambda3_key, shape=(self.pop_size // 2, self.dim))

        speed = state.speed
        new_speed = (
            lambda1 * speed[students]
            + lambda2 * (state.population[teachers] - state.population[students])
            + self.phi * lambda3 * (center - state.population[students])
        )
        new_population = state.population.at[students].add(new_speed)
        new_speed = speed.at[students].set(new_speed)

        new_population = jnp.clip(new_population, self.lb, self.ub)

        return state.update(population=new_population, speed=new_speed, key=key)
