from functools import partial

import jax
import jax.numpy as jnp
import copy

import evoxlib as exl


@exl.jit_class
class CSO(exl.Algorithm):
    def __init__(self, lb, ub, pop_size):
        self.dim = lb.shape[0]
        self.lb = lb
        self.ub = ub
        self.pop_size = pop_size

    def setup(self, key):
        state_key, init_key = jax.random.split(key)
        population = jax.random.uniform(init_key, shape=(self.pop_size, self.dim))
        population = population * (self.ub - self.lb) + self.lb
        speed = jnp.zeros(shape=(self.pop_size, self.dim))

        return exl.State(
            population=population,
            speed=speed,
            key=state_key
        )

    def ask(self, state):
        return state, state.population

    def tell(self, state, x, F):
        key = state.key
        key, subkey = jax.random.split(key)
        randperm = jax.random.permutation(subkey, self.pop_size).reshape(2, -1)
        mask = F[randperm[0, :]] < F[randperm[1, :]]

        teachers = jnp.where(mask, randperm[0, :], randperm[1, :])
        students = jnp.where(mask, randperm[1, :], randperm[0, :])

        key, subkey1, subkey2 = jax.random.split(key, num=3)
        lambda1 = jax.random.uniform(subkey1, shape=(self.pop_size // 2, self.dim))
        lambda2 = jax.random.uniform(subkey2, shape=(self.pop_size // 2, self.dim))

        speed = state.speed
        new_speed = lambda1 * speed[students] + lambda2 * (x[teachers] - x[students])
        new_population = x.at[students].add(new_speed)
        new_speed = speed.at[students].set(new_speed)

        return state.update(population=new_population, speed=new_speed, key=key)
