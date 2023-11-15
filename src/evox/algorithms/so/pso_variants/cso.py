# --------------------------------------------------------------------------------------
# 1. This code implements algorithms described in the following papers:
#
# Title: A Competitive Swarm Optimizer for Large Scale Optimization
# Link: https://ieeexplore.ieee.org/document/6819057
# --------------------------------------------------------------------------------------

import jax
import jax.numpy as jnp
from evox import Algorithm, State, jit_class


@jit_class
class CSO(Algorithm):
    def __init__(self, lb, ub, pop_size, phi=0, mean=None, stdev=None):
        self.dim = lb.shape[0]
        self.lb = lb
        self.ub = ub
        self.pop_size = pop_size
        self.phi = phi
        self.mean = mean
        self.stdev = stdev
        assert stdev is None or stdev > 0

    def setup(self, key):
        state_key, init_key = jax.random.split(key)
        if self.mean is not None and self.stdev is not None:
            population = self.stdev * jax.random.normal(
                init_key, shape=(self.pop_size, self.dim)
            )
            population = jnp.clip(population, self.lb, self.ub)
        else:
            population = jax.random.uniform(init_key, shape=(self.pop_size, self.dim))
            population = population * (self.ub - self.lb) + self.lb
        velocity = jnp.zeros((self.pop_size, self.dim))
        fitness = jnp.full((self.pop_size,), jnp.inf)

        return State(
            population=population,
            fitness=fitness,
            velocity=velocity,
            students=jnp.empty((self.pop_size // 2,), dtype=jnp.int32),
            key=state_key,
        )

    def init_ask(self, state):
        return state.population, state

    def init_tell(self, state, fitness):
        return state.update(fitness=fitness)

    def ask(self, state):
        key, pairing_key, lambda1_key, lambda2_key, lambda3_key = jax.random.split(
            state.key, num=5
        )
        randperm = jax.random.permutation(pairing_key, self.pop_size).reshape(2, -1)
        mask = state.fitness[randperm[0, :]] < state.fitness[randperm[1, :]]

        teachers = jnp.where(mask, randperm[0, :], randperm[1, :])
        students = jnp.where(mask, randperm[1, :], randperm[0, :])
        lambda1 = jax.random.uniform(lambda1_key, shape=(self.pop_size // 2, self.dim))
        lambda2 = jax.random.uniform(lambda2_key, shape=(self.pop_size // 2, self.dim))
        lambda3 = jax.random.uniform(lambda3_key, shape=(self.pop_size // 2, self.dim))
        center = jnp.mean(state.population, axis=0)
        student_velocity = (
            lambda1 * state.velocity[students]
            + lambda2 * (state.population[teachers] - state.population[students])
            + self.phi * lambda3 * (center - state.population[students])
        )
        candidates = jnp.clip(
            state.population[students] + student_velocity, self.lb, self.ub
        )
        new_population = state.population.at[students].set(candidates)
        new_velocity = state.velocity.at[students].set(student_velocity)

        return (
            candidates,
            state.update(
                population=new_population,
                velocity=new_velocity,
                students=students,
                key=key,
            ),
        )

    def tell(self, state, fitness):
        fitness = state.fitness.at[state.students].set(fitness)
        return state.update(fitness=fitness)
