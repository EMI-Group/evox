# --------------------------------------------------------------------------------------
# 1. This code implements algorithms described in the following papers:
#
# Title: Dynamic Multi-Swarm Particle Swarm Optimization Based on Elite Learning
# Link: https://ieeexplore.ieee.org/document/8936982
# --------------------------------------------------------------------------------------

import jax
import jax.numpy as jnp

from evox.utils import *
from evox import Algorithm, State, jit_class


# DMS-PSO-EL: dynamic multi-swarm particle swarm optimization based on an elite learning strategy
@jit_class
class DMSPSOEL(Algorithm):
    #: Population size N, dynamic sub-swarms size ND,
    # following sub-swarm size Nf,
    # number of dynamic sub-swarms ND_sub, size of each dynamic
    # sub-swarm SD_sub, and maximum number of
    # iterations Max_Iter
    def __init__(
        self,
        lb,  # lower bound of problem
        ub,  # upper bound of problem
        dynamic_sub_swarm_size,  # one of the dynamic sub-swarms size
        dynamic_sub_swarms_num,  # number of dynamic sub-swarms
        following_sub_swarm_size,  # following sub-swarm size
        regrouped_iteration_num,  # number of iterations for regrouping
        max_iteration,  # maximum number of iterations
        inertia_weight,  # w
        pbest_coefficient,  # c_pbest
        lbest_coefficient,  # c_lbest
        rbest_coefficient,  # c_rbest
        gbest_coefficient,  # c_gbest
    ):
        self.dim = lb.shape[0]
        self.lb = lb
        self.ub = ub
        self.pop_size = (
            dynamic_sub_swarm_size * dynamic_sub_swarms_num + following_sub_swarm_size
        )
        self.dynamic_sub_swarm_size = dynamic_sub_swarm_size
        self.dynamic_sub_swarms_num = dynamic_sub_swarms_num
        self.following_sub_swarm_size = following_sub_swarm_size
        self.regrouped_iteration_num = regrouped_iteration_num
        self.max_iteration = max_iteration
        self.w = inertia_weight
        self.c_pbest = pbest_coefficient
        self.c_lbest = lbest_coefficient
        self.c_rbest = rbest_coefficient
        self.c_gbest = gbest_coefficient

    def setup(self, key):
        state_key, init_pop_key, init_v_key = jax.random.split(key, 3)
        length = self.ub - self.lb
        population = jax.random.uniform(init_pop_key, shape=(self.pop_size, self.dim))
        population = population * length + self.lb

        dynamic_swarms = population[
            : self.dynamic_sub_swarm_size * self.dynamic_sub_swarms_num, :
        ]
        dynamic_swarms = dynamic_swarms.reshape(
            self.dynamic_sub_swarms_num, self.dynamic_sub_swarm_size, self.dim
        )

        following_swarm = population[
            self.dynamic_sub_swarm_size * self.dynamic_sub_swarms_num :, :
        ]

        lbest_position = dynamic_swarms[:, 0, :]
        lbest_fitness = jnp.full((self.dynamic_sub_swarms_num,), jnp.inf)

        velocity = jax.random.uniform(init_v_key, shape=(self.pop_size, self.dim))
        velocity = velocity * length * 2 - length

        return State(
            iteration=0,
            population=population,
            velocity=velocity,
            pbest_position=population,
            pbest_fitness=jnp.full((self.pop_size,), jnp.inf),
            lbest_position=lbest_position,
            lbest_fitness=lbest_fitness,
            rbest_index=jnp.zeros((self.following_sub_swarm_size,)).astype(jnp.int32),
            gbest_position=jnp.zeros((self.dim,)),
            gbest_fitness=jnp.inf,
            key=state_key,
        )

    def ask(self, state):
        return state.population, state

    def tell(self, state, fitness):
        state = jax.lax.cond(
            state.iteration < 0.9 * self.max_iteration,
            self._update_strategy_1,
            self._update_strategy_2,
            state,
            fitness,
        )

        return state

    def _update_strategy_1(self, state, fitness):
        state = jax.lax.cond(
            state.iteration % self.regrouped_iteration_num == 0,
            self._regroup,
            lambda x, y: x,
            state,
            fitness,
        )

        state_key, rand_key_pbest, rand_key_lbest, rand_key_rbest = jax.random.split(
            state.key, num=4
        )

        # ----------------- Update pbest -----------------
        compare = state.pbest_fitness > fitness
        pbest_position = jnp.where(
            compare[:, jnp.newaxis], state.population, state.pbest_position
        )
        pbest_fitness = jnp.minimum(state.pbest_fitness, fitness)

        # ----------------- dynamic swarms -----------------
        dynamic_swarms_position = state.population[
            : self.dynamic_sub_swarm_size * self.dynamic_sub_swarms_num, :
        ]
        dynamic_swarms_position = dynamic_swarms_position.reshape(
            self.dynamic_sub_swarms_num, self.dynamic_sub_swarm_size, self.dim
        )
        dynamic_swarms_fitness = fitness[
            : self.dynamic_sub_swarm_size * self.dynamic_sub_swarms_num
        ]
        dynamic_swarms_fitness = dynamic_swarms_fitness.reshape(
            self.dynamic_sub_swarms_num, self.dynamic_sub_swarm_size
        )
        dynamic_swarms_velocity = state.velocity[
            : self.dynamic_sub_swarm_size * self.dynamic_sub_swarms_num, :
        ]
        dynamic_swarms_velocity = dynamic_swarms_velocity.reshape(
            self.dynamic_sub_swarms_num, self.dynamic_sub_swarm_size, self.dim
        )
        dynamic_swarms_pbest = pbest_position[
            : self.dynamic_sub_swarm_size * self.dynamic_sub_swarms_num, :
        ]
        dynamic_swarms_pbest = dynamic_swarms_pbest.reshape(
            self.dynamic_sub_swarms_num, self.dynamic_sub_swarm_size, self.dim
        )

        # ----------------- following swarm -----------------
        following_swarm_position = state.population[
            self.dynamic_sub_swarm_size * self.dynamic_sub_swarms_num :, :
        ]
        following_swarm_fitness = fitness[
            self.dynamic_sub_swarm_size * self.dynamic_sub_swarms_num :
        ]
        following_swarm_velocity = state.velocity[
            self.dynamic_sub_swarm_size * self.dynamic_sub_swarms_num :, :
        ]
        following_swarm_pbest = pbest_position[
            self.dynamic_sub_swarm_size * self.dynamic_sub_swarms_num :, :
        ]

        # ----------------- Update lbest -----------------
        lbest_fitness = jnp.amin(
            dynamic_swarms_fitness, axis=1
        )  # shape:(dynamic_sub_swarms_num,)
        lbest_index = jnp.argmin(dynamic_swarms_fitness, axis=1)
        lbest_position = dynamic_swarms_position[
            jnp.arange(dynamic_swarms_position.shape[0]), lbest_index
        ]

        # ----------------- Update rbest -----------------
        rbest_position = state.population[state.rbest_index, :]

        # ---------------------------------------------------------
        # Caculate Dynamic Swarms Velocity
        rand_pbest = jax.random.uniform(rand_key_pbest, shape=(self.pop_size, self.dim))
        rand_lbest = jax.random.uniform(
            rand_key_lbest,
            shape=(self.dynamic_sub_swarms_num, self.dynamic_sub_swarm_size, self.dim),
        )

        dynamic_swarms_rand_pbest = rand_pbest[
            : self.dynamic_sub_swarm_size * self.dynamic_sub_swarms_num, :
        ]
        dynamic_swarms_rand_pbest = dynamic_swarms_rand_pbest.reshape(
            self.dynamic_sub_swarms_num, self.dynamic_sub_swarm_size, self.dim
        )
        dynamic_swarms_velocity = (
            self.w * dynamic_swarms_velocity
            + self.c_pbest
            * dynamic_swarms_rand_pbest
            * (dynamic_swarms_pbest - dynamic_swarms_position)
            + self.c_lbest
            * rand_lbest
            * (
                jnp.broadcast_to(
                    lbest_position[:, None, :],
                    shape=(
                        self.dynamic_sub_swarms_num,
                        self.dynamic_sub_swarm_size,
                        self.dim,
                    ),
                )
                - dynamic_swarms_position
            )  # broadcast
        )

        # ---------------------------------------------------------
        # Caculate Following Swarm Velocity
        folowing_swarm_rand_pbest = rand_pbest[
            self.dynamic_sub_swarm_size * self.dynamic_sub_swarms_num :, :
        ]
        rand_rbest = jax.random.uniform(
            rand_key_rbest, shape=(self.following_sub_swarm_size, self.dim)
        )

        following_swarm_velocity = (
            self.w * following_swarm_velocity
            + self.c_pbest
            * folowing_swarm_rand_pbest
            * (following_swarm_pbest - following_swarm_position)
            + self.c_rbest * rand_rbest * (rbest_position - following_swarm_position)
        )

        # ---------------------------------------------------------
        # Update Population
        dynamic_swarms_velocity = dynamic_swarms_velocity.reshape(
            self.dynamic_sub_swarm_size * self.dynamic_sub_swarms_num, self.dim
        )
        velocity = jnp.concatenate(
            (dynamic_swarms_velocity, following_swarm_velocity), axis=0
        )
        population = state.population + velocity
        population = jnp.clip(population, self.lb, self.ub)

        return state.update(
            iteration=state.iteration + 1,
            population=population,
            velocity=velocity,
            pbest_position=pbest_position,
            pbest_fitness=pbest_fitness,
            lbest_position=lbest_position,
            lbest_fitness=lbest_fitness,
            rbest_index=state.rbest_index,
            gbest_position=state.gbest_position,
            gbest_fitness=state.gbest_fitness,
            key=state_key,
        )

    def _regroup(self, state, fitness):
        sort_index = jnp.argsort(fitness, axis=0)
        dynamic_swarm_population_index = sort_index[
            : self.dynamic_sub_swarm_size * self.dynamic_sub_swarms_num
        ]
        state_key, shuffle_key = jax.random.split(state.key, num=2)
        dynamic_swarm_population_index = jax.random.permutation(
            shuffle_key, dynamic_swarm_population_index, independent=True
        )
        regroup_index = jnp.concatenate(
            (
                dynamic_swarm_population_index,
                sort_index[self.dynamic_sub_swarm_size * self.dynamic_sub_swarms_num :],
            ),
            axis=0,
        )

        population = state.population[regroup_index]
        velocity = state.velocity[regroup_index]
        pbest_position = state.pbest_position[regroup_index]
        pbest_fitness = state.pbest_fitness[regroup_index]

        dynamic_swarm_fitness = fitness[
            : self.dynamic_sub_swarm_size * self.dynamic_sub_swarms_num
        ]
        rbest_index = jnp.argsort(dynamic_swarm_fitness, axis=0)[
            : self.following_sub_swarm_size
        ]

        return state.update(
            iteration=state.iteration,
            population=population,
            velocity=velocity,
            pbest_position=pbest_position,
            pbest_fitness=pbest_fitness,
            lbest_position=state.lbest_position,
            lbest_fitness=state.lbest_fitness,
            rbest_index=rbest_index,
            gbest_position=state.gbest_position,
            gbest_fitness=state.gbest_fitness,
            key=state_key,
        )

    def _update_strategy_2(self, state, fitness):
        state_key, rand_key_pbest, rand_key_gbest = jax.random.split(state.key, num=3)

        # ---------------------------------------------------------
        # Update pbest
        compare = state.pbest_fitness > fitness
        pbest_position = jnp.where(
            compare[:, None], state.population, state.pbest_position
        )
        pbest_fitness = jnp.minimum(state.pbest_fitness, fitness)

        # ---------------------------------------------------------
        # Update gbest
        gbest_position = pbest_position[jnp.argmin(pbest_fitness, axis=0)]
        gbest_fitness = jnp.amin(pbest_fitness, axis=0)

        # ---------------------------------------------------------
        velocity = (
            self.w * state.velocity
            + self.c_pbest
            * jax.random.uniform(rand_key_pbest, shape=(self.pop_size, self.dim))
            * (pbest_position - state.population)
            + self.c_gbest
            * jax.random.uniform(rand_key_gbest, shape=(self.pop_size, self.dim))
            * (gbest_position - state.population)
        )
        population = state.population + velocity
        population = jnp.clip(population, self.lb, self.ub)

        return state.update(
            iteration=state.iteration + 1,
            population=population,
            velocity=velocity,
            pbest_position=pbest_position,
            pbest_fitness=pbest_fitness,
            lbest_position=state.lbest_position,
            lbest_fitness=state.lbest_fitness,
            rbest_index=state.rbest_index,
            gbest_position=gbest_position,
            gbest_fitness=gbest_fitness,
            key=state_key,
        )
