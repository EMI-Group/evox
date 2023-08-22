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


def manage_memories(i, success_failure_memory, strategy_ids, compare):
    # Fix strategy_ids and copmare for loop. Use partial().
    success_memory, failure_memory = success_failure_memory
    success_memory_up = success_memory.at[0, strategy_ids[i]].add(1)
    success_memory = lax.select(compare[i], success_memory_up, success_memory)

    failure_memory_up = failure_memory.at[0, strategy_ids[i]].add(1)
    failure_memory = lax.select(compare[i], failure_memory, failure_memory_up)

    return (success_memory, failure_memory)


def manage_CR_menory(i, CR_memory, strategy_ids, compare, CRs):
    str_idx = strategy_ids[i]
    CR = CRs[i]
    is_success = compare[i]

    CR_mk = CR_memory[:, str_idx]
    CR_mk_up = jnp.roll(CR_mk, shift=1)
    CR_mk_up = CR_mk_up.at[0].set(CR)
    CR_memory_up = lax.dynamic_update_slice_in_dim(
        CR_memory, CR_mk_up[:, jnp.newaxis], start_index=str_idx, axis=1
    )
    CR_memory = lax.select(is_success, CR_memory_up, CR_memory)
    return CR_memory


@jit_class
class SaDE(Algorithm):
    """SaDE
    Qin A K, Huang V L, Suganthan P N.
    Differential evolution algorithm with strategy adaptation for global numerical optimization[J].
    IEEE transactions on Evolutionary Computation, 2008, 13(2): 398-417.
    """

    def __init__(
        self,
        lb,
        ub,
        pop_size=100,
        diff_padding_num=9,
        LP=50,
    ):
        self.dim = lb.shape[0]
        self.lb = lb
        self.ub = ub
        self.pop_size = pop_size
        self.batch_size = pop_size

        self.diff_padding_num = diff_padding_num

        self.LP = LP
        # Strategy codes(4 bits): [basevect_prim, basevect_sec, diff_num, cross_strategy]
        # basevect: 0="rand", 1="best", 2="pbest", 3="current", cross_strategy: 0=bin, 1=exp, 2=arith
        rand_1_bin = jnp.array([0, 0, 1, 0])
        rand2best_2_bin = jnp.array([0, 1, 2, 0])
        rand_2_bin = jnp.array([0, 0, 2, 0])
        current2rand_1 = jnp.array([0, 0, 1, 2])  # current2rand_1 <==> rand_1_arith

        self.strategy_pool = jnp.stack(
            (rand_1_bin, rand2best_2_bin, rand_2_bin, current2rand_1)
        )

    def setup(self, key):
        state_key, init_key = jax.random.split(key, 2)
        population = jax.random.uniform(init_key, shape=(self.pop_size, self.dim))
        population = population * (self.ub - self.lb) + self.lb
        fitness = jnp.full((self.pop_size,), jnp.inf)
        trial_vectors = jnp.empty(shape=(self.batch_size, self.dim))
        best_index = 0
        start_index = 0
        success_memory = jnp.full(shape=(self.LP, 4), fill_value=0)
        failure_memory = jnp.full(shape=(self.LP, 4), fill_value=0)
        CR_memory = jnp.full(shape=(self.LP, 4), fill_value=jnp.nan)

        return State(
            population=population,
            fitness=fitness,
            best_index=best_index,
            start_index=start_index,
            key=state_key,
            trial_vectors=trial_vectors,
            success_memory=success_memory,
            failure_memory=failure_memory,
            CR_memory=CR_memory,
            CRs=jnp.empty(shape=(self.batch_size,)),
            strategy_ids=jnp.empty(shape=(self.batch_size,)),
            iter=0,
        )

    def ask(self, state):
        key, ask_one_key, strategy_key, CRs_key, CRs_key_repair = jax.random.split(
            state.key, 5
        )
        ask_one_keys = jax.random.split(ask_one_key, self.batch_size)
        indices = jnp.arange(self.batch_size) + state.start_index

        """Update strategy and Cr"""
        CRM_init = jnp.array([0.5, 0.5, 0.5, 0.5])
        strategy_p_init = jnp.array([0.25, 0.25, 0.25, 0.25])

        success_sum = jnp.sum(state.success_memory, axis=0)
        failure_sum = jnp.sum(state.failure_memory, axis=0)
        S_mat = (success_sum / (success_sum + failure_sum)) + 0.01
        strategy_p_update = S_mat / jnp.sum(S_mat)
        strategy_p = lax.select(
            state.iter >= self.LP, strategy_p_update, strategy_p_init
        )

        CRM_update = jnp.median(state.CR_memory, axis=0)
        CRM = lax.select(state.iter > self.LP, CRM_update, CRM_init)

        strategy_ids = jax.random.choice(
            strategy_key, a=4, shape=(self.batch_size,), replace=True, p=strategy_p
        )
        CRs_vect = jax.random.normal(CRs_key, shape=(self.batch_size, 4)) * 0.1 + CRM
        CRs_vect_repair = (
            jax.random.normal(CRs_key_repair, shape=(self.batch_size, 4)) * 0.1 + CRM
        )

        mask = (CRs_vect < 0) | (CRs_vect > 1)
        CRs_vect = jnp.where(mask, CRs_vect_repair, CRs_vect)

        trial_vectors, CRs = vmap(
            partial(
                self._ask_one,
                state_inner=state,
            )
        )(
            ask_one_key=ask_one_keys,
            index=indices,
            strategy_index=strategy_ids,
            CR=CRs_vect,
        )

        return trial_vectors, state.update(
            trial_vectors=trial_vectors,
            key=key,
            CRs=CRs,
            strategy_ids=strategy_ids,
            iter=state.iter + 1,
        )

    def _ask_one(self, state_inner, ask_one_key, index, strategy_index, CR):
        F_key, select_key, pbest_key, crossover_key = jax.random.split(ask_one_key, 4)

        population = state_inner.population
        best_index = state_inner.best_index
        fitness = state_inner.fitness

        # Reparameter
        differential_weight = jax.random.normal(F_key) * 0.3 + 0.5
        cross_probability = CR[strategy_index]

        strategy_code = self.strategy_pool[strategy_index]
        basevect_prim_type = strategy_code[0]
        basevect_sec_type = strategy_code[1]
        num_diff_vects = strategy_code[2]
        cross_strategy = strategy_code[3]

        difference_sum, rand_vect_idx = de_diff_sum(
            select_key,
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
            crossover_key,
            mutation_vector,
            current_vect,
            cross_probability,
        )

        trial_vector = jnp.clip(trial_vector, self.lb, self.ub)

        return trial_vector, CR[strategy_index]

    def tell(self, state, trial_fitness):
        start_index = state.start_index
        batch_pop = jax.lax.dynamic_slice_in_dim(
            state.population, start_index, self.batch_size, axis=0
        )
        batch_fitness = jax.lax.dynamic_slice_in_dim(
            state.fitness, start_index, self.batch_size, axis=0
        )

        compare = trial_fitness <= batch_fitness

        population_update = jnp.where(
            compare[:, jnp.newaxis], state.trial_vectors, batch_pop
        )
        fitness_update = jnp.where(compare, trial_fitness, batch_fitness)

        population = jax.lax.dynamic_update_slice_in_dim(
            state.population, population_update, start_index, axis=0
        )
        fitness = jax.lax.dynamic_update_slice_in_dim(
            state.fitness, fitness_update, start_index, axis=0
        )
        best_index = jnp.argmin(fitness)
        start_index = (state.start_index + self.batch_size) % self.pop_size

        """Update memories"""
        success_memory = jnp.roll(state.success_memory, shift=1, axis=0)
        success_memory = success_memory.at[0, :].set(0)
        failure_memory = jnp.roll(state.failure_memory, shift=1, axis=0)
        failure_memory = failure_memory.at[0, :].set(0)

        manage_memories_part = partial(
            manage_memories, strategy_ids=state.strategy_ids, compare=compare
        )
        success_memory, failure_memory = lax.fori_loop(
            0,
            self.pop_size,
            body_fun=manage_memories_part,
            init_val=(success_memory, failure_memory),
        )

        manage_CR_menory_part = partial(
            manage_CR_menory,
            strategy_ids=state.strategy_ids,
            CRs=state.CRs,
            compare=compare,
        )
        CR_memory = lax.fori_loop(
            0, self.pop_size, body_fun=manage_CR_menory_part, init_val=state.CR_memory
        )

        return state.update(
            population=population,
            fitness=fitness,
            best_index=best_index,
            start_index=start_index,
            success_memory=success_memory,
            failure_memory=failure_memory,
            CR_memory=CR_memory,
        )
