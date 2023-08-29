# --------------------------------------------------------------------------------------
# 1. This code implements algorithms described in the following papers:
#
# Title: A Cooperative approach to particle swarm optimization
# Link: https://ieeexplore.ieee.org/document/1304845
# --------------------------------------------------------------------------------------

import jax
import jax.numpy as jnp

from evox.utils import *
from evox import Algorithm, State, jit_class


# CPSO-S: Cooperative PSO
@jit_class
class CPSOS(Algorithm):
    def __init__(
        self,
        lb,  # lower bound of problem
        ub,  # upper bound of problem
        pop_size,  # population size for one swarm of a single dimension
        inertia_weight,  # w
        pbest_coefficient,  # c_pbest
        gbest_coefficient,  # c_gbest
    ):
        self.dim = lb.shape[0]
        self.lb = lb
        self.ub = ub
        self.pop_size = pop_size
        self.w = inertia_weight
        self.c_pbest = pbest_coefficient
        self.c_gbest = gbest_coefficient

    def setup(self, key):
        state_key, init_pop_key, init_v_key = jax.random.split(key, 3)
        ub = jnp.broadcast_to(self.ub[:, None], shape=(self.dim, self.pop_size))
        lb = jnp.broadcast_to(self.lb[:, None], shape=(self.dim, self.pop_size))
        length = ub - lb
        _population = jax.random.uniform(init_pop_key, shape=(self.dim, self.pop_size))
        _population = _population * length + lb

        context_vector = _population[:, 0]  # b
        broadcast_context_vector = jnp.broadcast_to(
            context_vector[:, None], (self.dim, self.pop_size)
        )
        cond = jnp.broadcast_to(
            jnp.arange(self.dim)[:, None] == jnp.arange(self.dim),
            shape=(self.dim, self.dim),
        )[:, :, None]
        population = jnp.where(
            cond, _population[None, :], broadcast_context_vector[None, :]
        )
        population = jnp.transpose(population, axes=(0, 2, 1))

        velocity = jax.random.uniform(init_v_key, shape=(self.dim, self.pop_size))
        velocity = velocity * length * 2 - length

        return State(
            # _population/velocity: shape:(dim, pop_size)
            # _population has dim different swarms, each swarm has pop_size particles
            # each particle in a swarm only records one single number, which is the position of this particle in this dimension
            _population=_population,
            velocity=velocity,
            # population: shape:(dim, pop_size, dim)
            # population has dim different swarms, each swarm has pop_size particles
            # population is the combination of _population and context_vector
            # it is used to calculate the fitness of each particle
            population=population,
            # pbest: like other algorithms, pbest is the best position of each particle
            pbest_position=_population,  # shape:(dim, pop_size)
            pbest_fitness=jnp.full(
                (
                    self.dim,
                    self.pop_size,
                ),
                jnp.inf,
            ),  # shape:(dim, pop_size)
            # gbest: !!! gbest is the best position of the swarm, not the whole population !!!
            # Because each swarm only focuses on one dimension, so the gbest in cpso_s should only record one single number,
            # But for the convenience of coding, we still use a array with shape(dim, dim) to represent the gbest position.
            # which represents the best position of this swarm.
            # therefore, the shape of gbest position is (dim, dim) and gbest fitness is (dim,)
            gbest_position=_population[:, 0],  # shape:(dim,)
            gbest_fitness=jnp.full((self.dim,), jnp.inf),  # shape:(dim,)
            # in fact, gbest_position is the same as context_vector
            # but we still use gbest_position to represent the best position of one swarm
            context_vector=context_vector,
            key=state_key,
        )

    def ask(self, state):
        return state.population, state

    # fitness: shape:(dim, pop_size)
    def tell(self, state, fitness):
        state_key, rand_key_gbest, rand_key_pbest = jax.random.split(state.key, num=3)

        # ----------------- Update pbest -----------------
        compare = state.pbest_fitness > fitness
        pbest_position = jnp.where(compare, state._population, state.pbest_position)
        pbest_fitness = jnp.minimum(state.pbest_fitness, fitness)

        # ----------------- Update gbest -----------------
        gbest_fitness = jnp.amin(pbest_fitness, axis=1)
        gbest_index = jnp.argmin(pbest_fitness, axis=1)
        gbest_position = pbest_position[
            jnp.arange(pbest_position.shape[0]), gbest_index
        ]

        # ------------------------------------------------------

        rand_pbest = jax.random.uniform(rand_key_pbest, shape=(self.dim, self.pop_size))
        rand_gbest = jax.random.uniform(rand_key_gbest, shape=(self.dim, self.pop_size))
        velocity = (
            self.w * state.velocity
            + self.c_pbest * rand_pbest * (pbest_position - state._population)
            + self.c_gbest
            * rand_gbest
            * (
                jnp.broadcast_to(
                    gbest_position[:, None], shape=(self.dim, self.pop_size)
                )
                - state._population
            )
        )
        _population = state._population + velocity
        ub = jnp.broadcast_to(self.ub[:, None], shape=(self.dim, self.pop_size))
        lb = jnp.broadcast_to(self.lb[:, None], shape=(self.dim, self.pop_size))
        _population = jnp.clip(_population, lb, ub)

        # ----------------- Update population -----------------
        context_vector = gbest_position
        broadcast_context_vector = jnp.broadcast_to(
            context_vector[:, None], (self.dim, self.pop_size)
        )
        cond = jnp.broadcast_to(
            jnp.arange(self.dim)[:, None] == jnp.arange(self.dim),
            shape=(self.dim, self.dim),
        )[:, :, None]
        population = jnp.where(
            cond, _population[None, :], broadcast_context_vector[None, :]
        )
        population = jnp.transpose(population, axes=(0, 2, 1))

        return state.update(
            _population=_population,
            velocity=velocity,
            population=population,
            pbest_position=pbest_position,
            pbest_fitness=pbest_fitness,
            gbest_position=gbest_position,
            gbest_fitness=gbest_fitness,
            context_vector=context_vector,
            key=state_key,
        )
