import jax
import jax.numpy as jnp
from jax import vmap, lax
from jax.experimental.host_callback import id_print

from evox import (
    Algorithm,
    Problem,
    State,
    algorithms,
    jit_class,
    monitors,
    pipelines,
    problems,
)
from evox.utils import *

"""Find the pbest vect"""
def find_pbest(fitness, pop_size, pbest_key, population):
        sorted_indices = jnp.argsort(fitness)
        top_p_num = int(pop_size * 0.05)   # The number is p,  ranges in 5% - 20%
        pbest_indices = sorted_indices[:top_p_num]
        pbest_index = jax.random.choice(pbest_key, pbest_indices)
        pbest_vect = population[pbest_index]
        return pbest_vect

"""Make differences and sum"""
def diff_sum(num_diff_vects, select_key, pop_size, diff_padding_num, index, population):
    # Randomly select 1 random individual (first index) and (num_diff_vects * 2) difference individuals
    dim = population.shape[1]
    select_len = num_diff_vects * 2 + 1
    random_choice = jax.random.choice(select_key, pop_size, shape=(diff_padding_num,), replace=False)
    random_choice = jnp.where(random_choice == index, pop_size - 1, random_choice)     # Ensure that selected indices != index

    # Permutate indices, take the first select_len individuals of indices in the population, and set the next individuals to 0
    pop_permut = population[random_choice]
    permut_mask = jnp.where(jnp.arange(diff_padding_num) < select_len, True, False)
    pop_permut_padding = jnp.where(permut_mask[:, jnp.newaxis], pop_permut, jnp.zeros((diff_padding_num, dim)))

    diff_vects = pop_permut_padding[1:, :]
    subtrahend_index = jnp.arange(1, diff_vects.shape[0], 2)  
    difference_sum = jnp.sum(diff_vects.at[subtrahend_index, :].multiply(-1), axis=0)

    rand_vect_idx = random_choice[0]
    return difference_sum, rand_vect_idx

"""These are three crossover strategies"""
def bin_cross(crossover_key, mutation_vector, current_vect, CR):
    # Binary crossover: dimension-by-dimension crossover
    # , based on cross_probability to determine the crossover needed for that dimension.
    dim = mutation_vector.shape[0]
    R_key, mask_key = jax.random.split(crossover_key)
    R = jax.random.choice(R_key, dim) # R is the jrand, i.e. the dimension that must be changed in the crossover
    mask = (jax.random.uniform(mask_key, shape=(dim,)) < CR)
    mask = mask.at[R].set(True) 
    
    trial_vector = jnp.where(mask, mutation_vector, current_vect,)           
    return trial_vector

def exp_cross(crossover_key, mutation_vector, current_vect, CR):
    # Exponential crossover: Cross the n-th to (n+l-1)-th dimension of the vector, 
    # and if n+l-1 exceeds the maximum dimension dim, then make it up from the beginning

    dim = mutation_vector.shape[0]
    n_key, l_key= jax.random.split(crossover_key)
    n = jax.random.choice(n_key, jnp.arange(dim))               

    # Generate l according to CR. n is the starting dimension to be crossover, and l is the crossover length
    l_mask = (jax.random.uniform(l_key, shape=(dim,)) < CR) 

    def count_forward_true(i, l, l_mask):              
        #count_forward_true() is used to count the number of preceding trues (randnum < cr)                                  
        replace_vect_init = jnp.arange(dim)
        replace_vect = jnp.where(replace_vect_init > i, True, False) 
        forward_mask = jnp.logical_or(replace_vect, l_mask)
        forward_bool = jnp.all(forward_mask)
        l = lax.select(forward_bool,  i+1, l)
        return l

    l = lax.fori_loop(0, dim, partial(count_forward_true, l_mask = l_mask), init_val = 0)
    # Generate mask by n and l
    mask_init = jnp.arange(dim)
    mask_bin = jnp.where(mask_init < l, True, False)                                    
    mask = jnp.roll(mask_bin, shift=n)
    trial_vector = jnp.where(mask, mutation_vector, current_vect,)
    return trial_vector

def arith_recom(crossover_key, mutation_vector, current_vect, K): 
    # K can take CR
    trial_vector = current_vect + K * (mutation_vector - current_vect)
    return trial_vector

"""Strategy codes(4 bits): [basevect_prim, basevect_sec, diff_num, cross_strategy] 
basevect: 0="rand", 1="best", 2="pbest", 3="current",
cross_strategy: 0=bin, 1=exp, 2=arith """
rand_1_bin = jnp.array([0, 0, 1, 0])
rand2best_2_bin = jnp.array([0, 1, 2, 0])
rand_2_bin = jnp.array([0, 0, 2, 0])
current2rand_1 = jnp.array([0, 0, 1, 2]) # current2rand_1 <==> rand_1_arith

current2pbest_1_bin = jnp.array([3, 2, 1, 0])

"""SaDE————————————————————————————————————————————————————————————————————————————————
Qin A K, Huang V L, Suganthan P N. Differential evolution algorithm with strategy adaptation for global numerical optimization[J]. 
IEEE transactions on Evolutionary Computation, 2008, 13(2): 398-417."""

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
        CR_memory, CR_mk_up[:,jnp.newaxis], start_index=str_idx, axis=1
        )
    CR_memory = lax.select(is_success, CR_memory_up, CR_memory)
    return CR_memory


@jit_class
class SaDE(Algorithm):
    def __init__(
        self,
        lb,
        ub,
        pop_size=100,
        diff_padding_num = 9,

        differential_weight=None,
        cross_probability=None,
        basevect_prim_type=None,
        basevect_sec_type=None,
        num_diff_vects=None,
        cross_strategy=None,
        LP = 50,
    ):
        self.num_diff_vects = num_diff_vects
        self.dim = lb.shape[0]
        self.lb = lb
        self.ub = ub
        self.pop_size = pop_size
        self.batch_size = pop_size      

        self.cross_probability = cross_probability
        self.differential_weight = differential_weight
        self.cross_strategy = cross_strategy
        self.diff_padding_num = diff_padding_num
        self.basevect_prim_type = basevect_prim_type
        self.basevect_sec_type = basevect_sec_type

        self.LP = LP
        self.strategy_pool = jnp.stack((rand_1_bin, rand2best_2_bin, rand_2_bin, current2rand_1))

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
            iter = 0,
        )

    def ask(self, state):
        key, ask_one_key, strategy_key, CRs_key, CRs_key_repair= jax.random.split(state.key, 5) 
        ask_one_keys = jax.random.split(ask_one_key, self.batch_size)
        indices = jnp.arange(self.batch_size) + state.start_index   

        """Update strategy and Cr"""
        CRM_init = jnp.array([0.5, 0.5, 0.5, 0.5])
        strategy_p_init = jnp.array([0.25, 0.25, 0.25, 0.25])

        success_sum = jnp.sum(state.success_memory, axis=0)
        failure_sum = jnp.sum(state.failure_memory, axis=0)
        S_mat = (success_sum / (success_sum + failure_sum)) + 0.01
        strategy_p_update = S_mat / jnp.sum(S_mat)
        strategy_p = lax.select(state.iter >= self.LP, strategy_p_update, strategy_p_init)

        CRM_update = jnp.median(state.CR_memory, axis=0)
        CRM = lax.select(state.iter > self.LP, CRM_update, CRM_init)

        strategy_ids = jax.random.choice(strategy_key, a=4, shape=(self.batch_size,), replace=True, p=strategy_p)
        CRs_vect = jax.random.normal(CRs_key, shape=(self.batch_size, 4)) * 0.1 + CRM
        CRs_vect_repair = jax.random.normal(CRs_key_repair, shape=(self.batch_size, 4)) * 0.1 + CRM

        mask = (CRs_vect < 0) | (CRs_vect > 1)
        CRs_vect = jnp.where(mask, CRs_vect_repair, CRs_vect)

        trial_vectors, CRs = vmap(partial(self._ask_one, state_inner = state,) )(ask_one_key=ask_one_keys, index=indices, strategy_index = strategy_ids, CR=CRs_vect)
        
        return trial_vectors, state.update(
            trial_vectors=trial_vectors, 
            key=key, CRs=CRs, 
            strategy_ids=strategy_ids,
            iter = state.iter + 1,
            ) 
    
    def _ask_one(self, state_inner, ask_one_key, index, strategy_index, CR):
        F_key, select_key, pbest_key, crossover_key = jax.random.split(ask_one_key, 4)

        population = state_inner.population
        best_index = state_inner.best_index
        fitness = state_inner.fitness

        # Reparameter
        self.differential_weight = jax.random.normal(F_key) * 0.3 + 0.5
        self.cross_probability = CR[strategy_index]

        strategy_code = self.strategy_pool[strategy_index]
        self.basevect_prim_type = strategy_code[0]
        self.basevect_sec_type = strategy_code[1]
        self.num_diff_vects = strategy_code[2]
        self.cross_strategy = strategy_code[3]
        
        difference_sum, rand_vect_idx = diff_sum(self.num_diff_vects, select_key, self.pop_size, self.diff_padding_num, index, population)
  
        rand_vect = population[rand_vect_idx]
        best_vect = population[best_index]     
        pbest_vect = find_pbest(fitness, self.pop_size, pbest_key, population)
        current_vect = population[index]
        vector_merge = jnp.stack((rand_vect, best_vect, pbest_vect, current_vect))

        base_vector_prim = vector_merge[self.basevect_prim_type]
        base_vector_sec = vector_merge[self.basevect_sec_type]

        base_vector = base_vector_prim + self.differential_weight * (base_vector_sec - base_vector_prim)

        mutation_vector = (base_vector + difference_sum * self.differential_weight)

        cross_funcs = (bin_cross, exp_cross, arith_recom)
        trial_vector = lax.switch(self.cross_strategy, cross_funcs, crossover_key, mutation_vector, current_vect, self.cross_probability)

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
        success_memory = success_memory.at[0,:].set(0)
        failure_memory = jnp.roll(state.failure_memory, shift=1, axis=0)
        failure_memory = failure_memory.at[0,:].set(0)

        manage_memories_part = partial(manage_memories, strategy_ids=state.strategy_ids, compare=compare)
        success_memory, failure_memory = lax.fori_loop(
            0, self.pop_size, body_fun=manage_memories_part, init_val=(success_memory, failure_memory)
            )

        manage_CR_menory_part = partial(manage_CR_menory, strategy_ids=state.strategy_ids, CRs=state.CRs, compare=compare)
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
            CR_memory = CR_memory,
        )


"""JaDE————————————————————————————————————————————————————————————————————————————————————————————————————
Zhang J, Sanderson A C. JADE: adaptive differential evolution with optional external archive[J]. 
IEEE Transactions on evolutionary computation, 2009, 13(5): 945-958."""

def get_success(i, S_F_CR, F_vect, CR_vect, compare):
    S_F, S_CR = S_F_CR
    is_success = compare[i]
    F = F_vect[i]
    CR = CR_vect[i]

    S_F_update_temp = jnp.roll(S_F, shift=1)
    S_F_update = S_F_update_temp.at[0].set(F)
    S_CR_update_temp = jnp.roll(S_CR, shift=1)
    S_CR_update = S_CR_update_temp.at[0].set(CR)

    S_F = lax.select(is_success, S_F_update, S_F_update_temp)
    S_CR = lax.select(is_success, S_CR_update, S_CR_update_temp)

    return (S_F, S_CR)


@jit_class
class JaDE(Algorithm):
    def __init__(
        self,
        lb,
        ub,
        pop_size=100,
        diff_padding_num = 9,

        differential_weight=None,
        cross_probability=None,
        basevect_prim_type=None,
        basevect_sec_type=None,
        num_diff_vects=None,
        cross_strategy=None,
        c = 0.1
    ):
        self.dim = lb.shape[0]
        self.lb = lb
        self.ub = ub
        self.pop_size = pop_size
        self.diff_padding_num = diff_padding_num
        self.batch_size = pop_size      
        self.c = c

        self.cross_probability = cross_probability
        self.differential_weight = differential_weight
        
        self.basevect_prim_type = current2pbest_1_bin[0]
        self.basevect_sec_type = current2pbest_1_bin[1]
        self.num_diff_vects = current2pbest_1_bin[2]
        self.cross_strategy = current2pbest_1_bin[3]

    def setup(self, key):  
        state_key, init_key = jax.random.split(key, 2)
        population = jax.random.uniform(init_key, shape=(self.pop_size, self.dim))
        population = population * (self.ub - self.lb) + self.lb
        fitness = jnp.full((self.pop_size,), jnp.inf)
        trial_vectors = jnp.zeros(shape=(self.batch_size, self.dim))
        best_index = 0
        start_index = 0  
        
        return State(
            population=population,
            fitness=fitness,
            best_index=best_index,
            start_index=start_index,
            key=state_key,
            trial_vectors=trial_vectors,
            F_u = 0.5,
            CR_u = 0.5,
            F_vect = jnp.empty(self.pop_size),
            CR_vect = jnp.empty(self.pop_size),
        )

    def ask(self, state):
        key, ask_one_key, F_key, CR_key = jax.random.split(state.key, 4) 
        ask_one_keys = jax.random.split(ask_one_key, self.batch_size)
        indices = jnp.arange(self.batch_size) + state.start_index    

        # Generare F and CR
        F_vect = jax.random.normal(F_key, shape=(self.pop_size,)) * 0.1 + state.F_u
        F_vect = jnp.clip(F_vect, jnp.zeros(self.pop_size), jnp.ones(self.pop_size))
        CR_vect = jax.random.cauchy(CR_key, shape=(self.pop_size,)) * 0.1 + state.CR_u
        CR_vect = jnp.clip(CR_vect, jnp.zeros(self.pop_size), jnp.ones(self.pop_size))
        
        trial_vectors = vmap(partial(self._ask_one, state_inner = state,) )(ask_one_key=ask_one_keys, index=indices, F=F_vect, CR=CR_vect)
        
        return trial_vectors, state.update(trial_vectors=trial_vectors, key=key, F_vect=F_vect, CR_vect=CR_vect) 
    
    def _ask_one(self, state_inner, ask_one_key, index, F, CR):
        select_key, pbest_key, crossover_key = jax.random.split(ask_one_key, 3)

        population = state_inner.population
        best_index = state_inner.best_index
        fitness = state_inner.fitness

        self.differential_weight = F
        self.cross_probability = CR
        
        difference_sum, rand_vect_idx = diff_sum(self.num_diff_vects, select_key, self.pop_size, self.diff_padding_num, index, population)
  
        rand_vect = population[rand_vect_idx]
        best_vect = population[best_index]     
        pbest_vect = find_pbest(fitness, self.pop_size, pbest_key, population)
        current_vect = population[index]
        vector_merge = jnp.stack((rand_vect, best_vect, pbest_vect, current_vect))

        base_vector_prim = vector_merge[self.basevect_prim_type]
        base_vector_sec = vector_merge[self.basevect_sec_type]

        base_vector = base_vector_prim + self.differential_weight * (base_vector_sec - base_vector_prim)

        mutation_vector = (base_vector + difference_sum * self.differential_weight)

        cross_funcs = (bin_cross, exp_cross, arith_recom)
        trial_vector = lax.switch(self.cross_strategy, cross_funcs, crossover_key, mutation_vector, current_vect, self.cross_probability)

        trial_vector = jnp.clip(trial_vector, self.lb, self.ub)
        
        return trial_vector

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

        """Update S_F and S_CR"""
        S_F_init = jnp.full(shape=(self.pop_size,), fill_value=jnp.nan)
        S_CR_init = jnp.full(shape=(self.pop_size,), fill_value=jnp.nan)

        get_success_part = partial(get_success, F_vect=state.F_vect, CR_vect=state.CR_vect, compare=compare)
        S_F, S_CR = lax.fori_loop(0, self.pop_size, body_fun=get_success_part, init_val=(S_F_init, S_CR_init))
        F_u = (1-self.c) * state.F_u + self.c * (jnp.nansum(S_F**2)/jnp.nansum(S_F))
        CR_u = (1-self.c) * state.CR_u + self.c * jnp.nanmean(S_CR)

        return state.update(
            population=population,
            fitness=fitness,
            best_index=best_index,
            start_index=start_index,
            F_u = F_u,
            CR_u = CR_u,
        )

"""CoDE"————————————————————————————————————————————————————————————————————————————————————————————————
Wang Y, Cai Z, Zhang Q. Differential evolution with composite trial vector generation strategies and control parameters[J]. 
IEEE transactions on evolutionary computation, 2011, 15(1): 55-66."""

@jit_class
class CoDE(Algorithm):
    def __init__(
        self,
        lb,
        ub,
        pop_size=100,
        diff_padding_num = 9,

        differential_weight=None,
        cross_probability=None,
        basevect_prim_type=None,
        basevect_sec_type=None,
        num_diff_vects=None,
        cross_strategy=None,

        param_pool = jnp.array([[1, 0.1], [1, 0.9], [0.8, 0.2]])
    ):
        self.num_diff_vects = num_diff_vects
        self.dim = lb.shape[0]
        self.lb = lb
        self.ub = ub
        self.pop_size = pop_size
        self.batch_size = pop_size      
        self.cross_probability = cross_probability
        self.differential_weight = differential_weight
        self.cross_strategy = cross_strategy
        self.diff_padding_num = diff_padding_num
        self.basevect_prim_type = basevect_prim_type
        self.basevect_sec_type = basevect_sec_type
        self.param_pool = param_pool
        

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
        ask_one_keys = jax.random.split(ask_one_key, self.batch_size)
        indices = jnp.arange(self.batch_size) + state.start_index    

        param_key_1, param_key_2, param_key_3 = jax.random.split(param_key, 3) 

        ## rand/1/bin + randomly selected param setting
        self.basevect_prim_type = rand_1_bin[0]
        self.basevect_sec_type = rand_1_bin[1]
        self.num_diff_vects = rand_1_bin[2]
        self.cross_strategy = rand_1_bin[3]
        param_ids = jax.random.choice(param_key_1, a=3, shape=(self.batch_size,), replace=True)
        trial_vectors_1 = vmap(partial(self._ask_one, state_inner = state,) )(ask_one_key=ask_one_keys, index=indices, param_idx=param_ids)

        ## rand/2/bin + randomly selected param setting
        self.basevect_prim_type = rand_2_bin[0]
        self.basevect_sec_type = rand_2_bin[1]
        self.num_diff_vects = rand_2_bin[2]
        self.cross_strategy = rand_2_bin[3]
        param_ids = jax.random.choice(param_key_2, a=3, shape=(self.batch_size,), replace=True)
        trial_vectors_2 = vmap(partial(self._ask_one, state_inner = state,) )(ask_one_key=ask_one_keys, index=indices, param_idx=param_ids)

        ## current2rand/1 + randomly selected param setting
        self.basevect_prim_type = current2rand_1[0]
        self.basevect_sec_type = current2rand_1[1]
        self.num_diff_vects = current2rand_1[2]
        self.cross_strategy = current2rand_1[3]
        param_ids = jax.random.choice(param_key_3, a=3, shape=(self.batch_size,), replace=True)
        trial_vectors_3 = vmap(partial(self._ask_one, state_inner = state,) )(ask_one_key=ask_one_keys, index=indices, param_idx=param_ids)

        trial_vectors = jnp.concatenate((trial_vectors_1, trial_vectors_2, trial_vectors_3), axis=0)
        
        return trial_vectors, state.update(trial_vectors=trial_vectors, key=key) 
    
    def _ask_one(self, state_inner, ask_one_key, index, param_idx):
        select_key, pbest_key, crossover_key = jax.random.split(ask_one_key, 3)

        population = state_inner.population
        best_index = state_inner.best_index
        fitness = state_inner.fitness

        params = self.param_pool[param_idx]
        self.differential_weight = params[0]
        self.cross_probability = params[1]
        
        difference_sum, rand_vect_idx = diff_sum(self.num_diff_vects, select_key, self.pop_size, self.diff_padding_num, index, population)
  
        rand_vect = population[rand_vect_idx]
        best_vect = population[best_index]     
        pbest_vect = find_pbest(fitness, self.pop_size, pbest_key, population)
        current_vect = population[index]
        vector_merge = jnp.stack((rand_vect, best_vect, pbest_vect, current_vect))

        base_vector_prim = vector_merge[self.basevect_prim_type]
        base_vector_sec = vector_merge[self.basevect_sec_type]

        base_vector = base_vector_prim + self.differential_weight * (base_vector_sec - base_vector_prim)

        mutation_vector = (base_vector + difference_sum * self.differential_weight)

        cross_funcs = (bin_cross, exp_cross, arith_recom)
        trial_vector = lax.switch(self.cross_strategy, cross_funcs, crossover_key, mutation_vector, current_vect, self.cross_probability)

        trial_vector = jnp.clip(trial_vector, self.lb, self.ub)
        
        return trial_vector

    def tell(self, state, trial_fitness):   
        # Compare the best of the 3 corresponding individuals in the 3 fitness vectors
        indices = jnp.arange(3*self.pop_size).reshape(3, self.pop_size)
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


"""SHADE————————————————————————————————————————————————————————————————————————————————————————————————————
Tanabe R, Fukunaga A. Success-history based parameter adaptation for differential evolution[C]//2013 
IEEE congress on evolutionary computation. IEEE, 2013: 71-78."""

def get_success_delta(i, S_F_CR_delta, F_vect, CR_vect, compare, deltas):
    S_F, S_CR, S_delta= S_F_CR_delta
    is_success = compare[i]
    F = F_vect[i]
    CR = CR_vect[i]
    delta = deltas[i]

    S_F_update_temp = jnp.roll(S_F, shift=1)
    S_F_update = S_F_update_temp.at[0].set(F)
    S_CR_update_temp = jnp.roll(S_CR, shift=1)
    S_CR_update = S_CR_update_temp.at[0].set(CR)
    S_delta_update_temp = jnp.roll(S_delta, shift=1)
    S_delta_update = S_delta_update_temp.at[0].set(delta)

    S_F = lax.select(is_success, S_F_update, S_F_update_temp)
    S_CR = lax.select(is_success, S_CR_update, S_CR_update_temp)
    S_delta = lax.select(is_success, S_delta_update, S_delta_update_temp)

    return (S_F, S_CR, S_delta)

@jit_class
class SHADE(Algorithm):
    def __init__(
        self,
        lb,
        ub,
        pop_size=100,
        diff_padding_num = 9,

        differential_weight=None,
        cross_probability=None,
        basevect_prim_type=None,
        basevect_sec_type=None,
        num_diff_vects=None,
        cross_strategy=None,
    ):
        self.dim = lb.shape[0]
        self.lb = lb
        self.ub = ub
        self.pop_size = pop_size
        self.diff_padding_num = diff_padding_num
        self.batch_size = pop_size      
        self.H = pop_size

        self.cross_probability = cross_probability
        self.differential_weight = differential_weight
        
        self.basevect_prim_type = current2pbest_1_bin[0]
        self.basevect_sec_type = current2pbest_1_bin[1]
        self.num_diff_vects = current2pbest_1_bin[2]
        self.cross_strategy = current2pbest_1_bin[3]

    def setup(self, key):  
        state_key, init_key = jax.random.split(key, 2)
        population = jax.random.uniform(init_key, shape=(self.pop_size, self.dim))
        population = population * (self.ub - self.lb) + self.lb
        fitness = jnp.full((self.pop_size,), jnp.inf)
        trial_vectors = jnp.zeros(shape=(self.batch_size, self.dim))
        best_index = 0
        start_index = 0 
        
        return State(
            population=population,
            fitness=fitness,
            best_index=best_index,
            start_index=start_index,
            key=state_key,
            trial_vectors=trial_vectors,
            Memory_FCR = jnp.full(shape=(2, 100), fill_value=0.5),
            F_vect = jnp.empty(self.pop_size),
            CR_vect = jnp.empty(self.pop_size),
        )

    def ask(self, state):
        key, ask_one_key, choice_key, F_key, CR_key = jax.random.split(state.key, 5)
        ask_one_keys = jax.random.split(ask_one_key, self.batch_size)
        indices = jnp.arange(self.batch_size) + state.start_index

        FCR_ids = jax.random.choice(choice_key, a=self.H, shape=(self.batch_size,), replace=True)
        M_F_vect = state.Memory_FCR[0, FCR_ids]
        M_CR_vect = state.Memory_FCR[1, FCR_ids]

        # Generare F and CR
        F_vect = jax.random.normal(F_key, shape=(self.pop_size,)) * 0.1 + M_F_vect
        F_vect = jnp.clip(F_vect, jnp.zeros(self.pop_size), jnp.ones(self.pop_size))

        CR_vect = jax.random.cauchy(CR_key, shape=(self.pop_size,)) * 0.1 + M_CR_vect
        CR_vect = jnp.clip(CR_vect, jnp.zeros(self.pop_size), jnp.ones(self.pop_size))
        
        trial_vectors = vmap(partial(self._ask_one, state_inner = state,) )(ask_one_key=ask_one_keys, index=indices, F=F_vect, CR=CR_vect)
        
        return trial_vectors, state.update(trial_vectors=trial_vectors, key=key, F_vect=F_vect, CR_vect=CR_vect) 
    
    def _ask_one(self, state_inner, ask_one_key, index, F, CR):
        select_key, pbest_key, crossover_key = jax.random.split(ask_one_key, 3)

        population = state_inner.population
        best_index = state_inner.best_index
        fitness = state_inner.fitness

        self.differential_weight = F
        self.cross_probability = CR
        
        difference_sum, rand_vect_idx = diff_sum(self.num_diff_vects, select_key, self.pop_size, self.diff_padding_num, index, population)
  
        rand_vect = population[rand_vect_idx]
        best_vect = population[best_index]     
        pbest_vect = find_pbest(fitness, self.pop_size, pbest_key, population)
        current_vect = population[index]
        vector_merge = jnp.stack((rand_vect, best_vect, pbest_vect, current_vect))

        base_vector_prim = vector_merge[self.basevect_prim_type]
        base_vector_sec = vector_merge[self.basevect_sec_type]

        base_vector = base_vector_prim + self.differential_weight * (base_vector_sec - base_vector_prim)

        mutation_vector = (base_vector + difference_sum * self.differential_weight)

        cross_funcs = (bin_cross, exp_cross, arith_recom)
        trial_vector = lax.switch(self.cross_strategy, cross_funcs, crossover_key, mutation_vector, current_vect, self.cross_probability)

        trial_vector = jnp.clip(trial_vector, self.lb, self.ub)
        
        return trial_vector

    def tell(self, state, trial_fitness):   
        start_index = state.start_index
        batch_pop = jax.lax.dynamic_slice_in_dim( 
            state.population, start_index, self.batch_size, axis=0
        )
        batch_fitness = jax.lax.dynamic_slice_in_dim(
            state.fitness, start_index, self.batch_size, axis=0
        )

        compare = trial_fitness < batch_fitness 

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

        """Update S_F and S_CR"""
        S_F_init = jnp.full(shape=(self.pop_size,), fill_value=jnp.nan)
        S_CR_init = jnp.full(shape=(self.pop_size,), fill_value=jnp.nan)
        S_delta_init = jnp.full(shape=(self.pop_size,), fill_value=jnp.nan)

        deltas = batch_fitness - trial_fitness
        get_success_part = partial(get_success_delta, F_vect=state.F_vect, CR_vect=state.CR_vect, compare=compare, deltas=deltas)
        S_F, S_CR, S_delta = lax.fori_loop(0, self.pop_size, body_fun=get_success_part, init_val=(S_F_init, S_CR_init, S_delta_init))

        norm_delta = S_delta / jnp.nansum(S_delta)
        M_CR = jnp.nansum(norm_delta * S_CR)
        M_F = jnp.nansum(norm_delta * (S_F**2)) / jnp.nansum(norm_delta * S_F)
        
        Memory_FCR_update = jnp.roll(state.Memory_FCR, shift=1, axis=1)
        Memory_FCR_update = Memory_FCR_update.at[0, 0].set(M_F)
        Memory_FCR_update = Memory_FCR_update.at[1, 0].set(M_CR)

        is_F_nan = jnp.isnan(M_F)
        Memory_FCR_update = lax.select(is_F_nan, state.Memory_FCR, Memory_FCR_update)

        is_S_nan = jnp.all(jnp.isnan(compare))
        Memory_FCR = lax.select(is_S_nan, state.Memory_FCR, Memory_FCR_update)

        return state.update(
            population=population,
            fitness=fitness,
            best_index=best_index,
            start_index=start_index,
            Memory_FCR=Memory_FCR,
        )