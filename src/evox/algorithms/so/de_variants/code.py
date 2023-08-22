import jax
from jax import lax, vmap
import jax.numpy as jnp
from evox import Algorithm, jit_class, State
from evox.operators.selection import select_rand_pbest
from evox.operators.crossover import (
    de_diff_sum,
    de_arith_recom,
    de_bin_cross,
    de_exp_cross,
)

from functools import partial


"""Strategy codes(4 bits): [basevect_prim, basevect_sec, diff_num, cross_strategy]
basevect: 0="rand", 1="best", 2="pbest", 3="current",
cross_strategy: 0=bin, 1=exp, 2=arith """
rand_1_bin = jnp.array([0, 0, 1, 0])
rand2best_2_bin = jnp.array([0, 1, 2, 0])
rand_2_bin = jnp.array([0, 0, 2, 0])
current2rand_1 = jnp.array([0, 0, 1, 2])  # current2rand_1 <==> rand_1_arith

current2pbest_1_bin = jnp.array([3, 2, 1, 0])


@jit_class
class CoDE(Algorithm):
    """CoDE
    Wang Y, Cai Z, Zhang Q.
    Differential evolution with composite trial vector generation strategies and control parameters[J].
    IEEE transactions on evolutionary computation, 2011, 15(1): 55-66.
    """

    def __init__(
        self,
        lb,
        ub,
        pop_size=100,
        diff_padding_num=9,
        param_pool=jnp.array([[1, 0.1], [1, 0.9], [0.8, 0.2]]),
    ):
        self.dim = lb.shape[0]
        self.lb = lb
        self.ub = ub
        self.pop_size = pop_size
        self.batch_size = pop_size
        self.param_pool = param_pool
        self.diff_padding_num = diff_padding_num
        self.strategies = jnp.array([rand_1_bin, rand_2_bin, current2rand_1])

    def setup(self, key):
        state_key, init_key = jax.random.split(key, 2)
        population = jax.random.uniform(init_key, shape=(self.pop_size, self.dim))
        population = population * (self.ub - self.lb) + self.lb
        fitness = jnp.full((self.pop_size,), jnp.inf)
        trial_vectors = jnp.zeros(shape=(self.batch_size * 3, self.dim))
        best_index = 0
        start_index = 0

        return State(
            population=population,
            fitness=fitness,
            best_index=best_index,
            start_index=start_index,
            key=state_key,
            trial_vectors=trial_vectors,
        )

    def ask(self, state):
        key, ask_one_key, param_key = jax.random.split(state.key, 3)

        ask_one_keys = jax.random.split(ask_one_key, 3 * self.batch_size).reshape(
            3, self.batch_size, 2
        )
        indices = jnp.arange(self.batch_size) + state.start_index

        param_keys = jax.random.split(param_key, 3)
        trial_vectors = []

        # run 3 different strategies
        for strategy, param_key, ask_one_key in zip(
            self.strategies, param_keys, ask_one_keys
        ):
            param_ids = jax.random.choice(
                param_key, a=3, shape=(self.batch_size,), replace=True
            )
            trial_vectors.append(
                vmap(partial(self._ask_one, state, strategy))(
                    ask_one_key, indices, param_ids
                )
            )

        trial_vectors = jnp.concatenate(trial_vectors, axis=0)

        return trial_vectors, state.update(trial_vectors=trial_vectors, key=key)

    def _ask_one(self, state, strategy, key, index, param_idx):
        diff_sum_key, pbest_key, recom_key = jax.random.split(key, 3)
        basevect_prim_type = strategy[0]
        basevect_sec_type = strategy[1]
        num_diff_vects = strategy[2]
        cross_strategy = strategy[3]

        population = state.population
        best_index = state.best_index
        fitness = state.fitness

        params = self.param_pool[param_idx]
        differential_weight = params[0]
        cross_probability = params[1]

        difference_sum, rand_vect_idx = de_diff_sum(
            diff_sum_key,
            self.diff_padding_num,
            num_diff_vects,
            index,
            population,
        )

        rand_vect = population[rand_vect_idx]
        best_vect = population[best_index]
        pbest_vect = select_rand_pbest(pbest_key, 0.05, population, fitness)
        current_vect = population[index]
        vector_merge = jnp.stack((rand_vect, best_vect, pbest_vect, current_vect))

        base_vector_prim = vector_merge[basevect_prim_type]
        base_vector_sec = vector_merge[basevect_sec_type]

        base_vector = base_vector_prim + differential_weight * (
            base_vector_sec - base_vector_prim
        )

        mutation_vector = base_vector + difference_sum * differential_weight

        cross_funcs = (
            de_bin_cross,
            de_exp_cross,
            lambda _key, x, y, z: de_arith_recom(x, y, z),
        )
        trial_vector = lax.switch(
            cross_strategy,
            cross_funcs,
            recom_key,
            mutation_vector,
            current_vect,
            cross_probability,
        )

        trial_vector = jnp.clip(trial_vector, self.lb, self.ub)

        return trial_vector

    def tell(self, state, trial_fitness):
        # Compare the best of the 3 corresponding individuals in the 3 fitness vectors
        indices = jnp.arange(3 * self.pop_size).reshape(3, self.pop_size)
        trans_fit = trial_fitness[indices]
        min_indices = jnp.argmin(trans_fit, axis=0)
        # min_indices_global is the corresponding best position found by flattening the three fitness vectors
        min_indices_global = indices[min_indices, jnp.arange(self.pop_size)]
        trial_fitness_select = trial_fitness[min_indices_global]
        trial_vectors_select = state.trial_vectors[min_indices_global]

        start_index = state.start_index
        batch_pop = jax.lax.dynamic_slice_in_dim(
            state.population, start_index, self.batch_size, axis=0
        )
        batch_fitness = jax.lax.dynamic_slice_in_dim(
            state.fitness, start_index, self.batch_size, axis=0
        )

        compare = trial_fitness_select <= batch_fitness

        population_update = jnp.where(
            compare[:, jnp.newaxis], trial_vectors_select, batch_pop
        )
        fitness_update = jnp.where(compare, trial_fitness_select, batch_fitness)

        population = jax.lax.dynamic_update_slice_in_dim(
            state.population, population_update, start_index, axis=0
        )
        fitness = jax.lax.dynamic_update_slice_in_dim(
            state.fitness, fitness_update, start_index, axis=0
        )
        best_index = jnp.argmin(fitness)
        start_index = (state.start_index + self.batch_size) % self.pop_size
        return state.update(
            population=population,
            fitness=fitness,
            best_index=best_index,
            start_index=start_index,
        )
