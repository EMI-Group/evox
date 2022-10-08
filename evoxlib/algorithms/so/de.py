import jax
import jax.numpy as jnp
from jax import lax
from enum import Enum

import evoxlib as exl


class MutationVectorTypeEnum(Enum):
    RAND_1 = 0
    BEST_2 = 1


@exl.jit_class
class DE(exl.Algorithm):
    def __init__(self, lb, ub, pop_size, batch_size, scale_factor=0.2, cross_rate=0.5, mutation_type=MutationVectorTypeEnum.RAND_1):
        assert(jnp.all(lb < ub))
        assert(pop_size >= 5) # otherwise the batch_size assert must fail
        assert(mutation_type >= MutationVectorTypeEnum.RAND_1 and mutation_type <= MutationVectorTypeEnum.BEST_2)
        if mutation_type == MutationVectorTypeEnum.RAND_1:
            assert(batch_size >= 1 and batch_size <= pop_size // 4)
        elif mutation_type == MutationVectorTypeEnum.BEST_2:
            assert(batch_size >= 1 and batch_size <= pop_size // 5)
        assert(scale_factor > 0 and scale_factor < 1)
        assert(cross_rate > 0 and cross_rate < 1)

        self.dim = lb.shape[0]
        self.lb = lb
        self.ub = ub
        self.pop_size = pop_size
        self.batch_size = batch_size
        self.scale_factor = scale_factor
        self.cross_rate = cross_rate
        self.mutation_type = mutation_type

    def setup(self, key):
        state_key, init_key = jax.random.split(key)
        population = jax.random.uniform(init_key, shape=(self.pop_size, self.dim))
        population = population * (self.ub - self.lb) + self.lb

        return exl.State(population=population, key=state_key)

    def ask(self, state):
        return state, state.population

    def tell(self, state, fitness):
        key, selection_key, crossover_key, preserve_key = jax.random.split(state.key, num=4)
        shuffle_pop = jax.random.shuffle(selection_key, state.population)
        if self.mutation_type == MutationVectorTypeEnum.RAND_1:
            new_pop = shuffle_pop[self.batch_size : self.batch_size * 2] + self.scale_factor *\
                        (shuffle_pop[self.batch_size * 2 : self.batch_size * 3] - shuffle_pop[self.batch_size * 3 : self.batch_size * 4])
        elif self.mutation_type == MutationVectorTypeEnum.BEST_2:
            best_idx = jnp.argmin(fitness)
            best = state.population[best_idx]
            new_pop = best + self.scale_factor * (shuffle_pop[self.batch_size * 1 : self.batch_size * 2] + shuffle_pop[self.batch_size * 2 : self.batch_size * 3] \
                                                - shuffle_pop[self.batch_size * 3 : self.batch_size * 4] - shuffle_pop[self.batch_size * 4 : self.batch_size * 5])
        old_pop = shuffle_pop[:self.batch_size]
        rand_crossover = jax.random.uniform(crossover_key, shape=(self.batch_size, self.dim))
        rand_preserve = jax.random.randint(preserve_key, self.batch_size, minval=0, maxval=self.dim)
        rand_crossover = lax.map(lambda c, p: c.at[p].set(0), zip(rand_crossover, rand_preserve))
        mask = rand_crossover <= self.cross_rate
        new_pop = jnp.where(mask, new_pop, old_pop)
    
        new_pop = jnp.clip(new_pop, self.lb, self.ub)

        return state.update(population=jnp.concatenate([new_pop, shuffle_pop[self.batch_size:]]), key=key)
