# R. Cheng, Y. Jin, K. Narukawa, and B. Sendhoff, A multiobjective evolutionary algorithm using Gaussian process-based inverse modeling,
# IEEE Transactions on Evolutionary Computation, 2015, 19(6): 838-856.


import jax
import jax.numpy as jnp
from jax import jit, vmap, random
from evox import Algorithm, jit_class, State
from evox.operators import (
    non_dominated_sort,
    crowding_distance,
    selection,
    mutation,
)
from evox.utils import cos_dist
from evox.operators.gaussian_process import GPRegression
import optax as ox
import jax.lax as lax
from evox.operators.sampling import UniformSampling
from evox.operators.selection import non_dominate

try:
    from gpjax.kernels import Linear
    from gpjax.likelihoods import Gaussian
except ImportError as e:
    original_error_msg = str(e)

    def Linear(*args, **kwargs):
        raise ImportError(
            f'Linear requires gpjax, but got "{original_error_msg}" when importing'
        )

    def Gaussian(*args, **kwargs):
        raise ImportError(
            f'Gaussian requires gpjax, but got "{original_error_msg}" when importing'
        )


# jax.config.update('jax_enable_x64', True)
def random_fill(key, matrix, matrix2, mask):
    indices = jnp.arange(matrix.shape[0])
    masked_indices = jnp.where(mask, indices, jnp.inf)
    sorted_indices = jnp.sort(masked_indices)
    max_indices = jnp.sum(mask)
    random_array = random.randint(
        key=key, shape=(matrix.shape[0],), minval=0, maxval=max_indices
    )
    new_indices = sorted_indices[random_array].astype(jnp.int32)
    matrix = matrix[new_indices, :]
    matrix2 = matrix2[new_indices, :]

    return matrix, matrix2


class IMMOEA(Algorithm):
    def __init__(
        self,
        lb,
        ub,
        n_objs=3,
        pop_size=105,
        l=3,
        k=10,
        mutation_op=None,
    ):
        self.lb = lb
        self.ub = ub
        self.n_objs = n_objs
        self.dim = lb.shape[0]
        self.pop_size = pop_size
        self.l = l
        self.k = k
        self.mutation = mutation_op

        if self.mutation is None:
            self.mutation = mutation.Polynomial((self.lb, self.ub))

    # initial set
    def setup(self, key):
        key, subkey = jax.random.split(key)
        self.pop_size = int(jnp.ceil(self.pop_size / self.k) * self.k)
        W = UniformSampling(self.k, self.n_objs)()[0]
        W = jnp.fliplr(jnp.sort(jnp.fliplr(W), axis=1))
        population = (
            jax.random.uniform(subkey, shape=(self.pop_size, self.dim))
            * (self.ub - self.lb)
            + self.lb
        )
        return State(
            population=population,
            fitness=jnp.zeros((self.pop_size, self.n_objs)),
            next_generation=population,
            key=key,
            reference_vector=W,
        )

    # generate initial population
    def init_ask(self, state):
        return state.population, state

    def init_tell(self, state, fitness):
        state = state.update(fitness=fitness)
        return state

    # generate next generation
    def ask(self, state):
        population = state.population
        fitness = state.fitness
        key, key1 = jax.random.split(state.key)
        distances = cos_dist(fitness, state.reference_vector)
        partition = jnp.argmax(distances, axis=1)

        def get_sub_pop(i):
            mask = partition == i
            sub_pop, _ = self.gp_offspring_(state, population, mask, fitness)
            return sub_pop

        sub_pops = jax.vmap(get_sub_pop, out_axes=0)(jnp.arange(self.k))
        next_generation = jnp.vstack(sub_pops)
        nan_mask = jnp.isnan(next_generation).sum(axis=1).astype(jnp.bool_)
        indices = jnp.arange(next_generation.shape[0])
        masked_indices = jnp.where(nan_mask, jnp.inf, indices)
        sorted_indices = jnp.sort(masked_indices).astype(jnp.int32)
        next_generation = next_generation[sorted_indices, :]
        next_generation = lax.dynamic_slice(next_generation, (0, 0), population.shape)
        next_generation = jnp.clip(next_generation, self.lb, self.ub)
        return next_generation, state.update(next_generation=next_generation, key=key)

    def tell(self, state, fitness):
        merged_pop = jnp.concatenate([state.population, state.next_generation], axis=0)
        merged_fitness = jnp.concatenate([state.fitness, fitness], axis=0)

        rank = non_dominated_sort(merged_fitness)
        order = jnp.argsort(rank)
        worst_rank = rank[order[self.pop_size]]
        mask = rank == worst_rank
        crowding_dis = crowding_distance(merged_fitness, mask)

        combined_order = jnp.lexsort((-crowding_dis, rank))[: self.pop_size]
        survivor = merged_pop[combined_order]
        survivor_fitness = merged_fitness[combined_order]
        state = state.update(population=survivor, fitness=survivor_fitness)
        return state

    # select next generation, but add clustering. The effect is not good, but it is the IM-MOEA's selection in PlatEMO.
    def another_tell(self, state, fitness):
        merged_pop = jnp.concatenate([state.population, state.next_generation], axis=0)
        merged_fitness = jnp.concatenate([state.fitness, fitness], axis=0)
        std_size = jnp.ceil(self.pop_size / self.k)
        distances = cos_dist(merged_fitness, state.reference_vector)
        partition = jnp.argmax(distances, axis=1)
        class_num, _ = self.get_class_num(state, partition)
        all_sub_pop = []
        all_sub_pop_fit = []

        def handle_small_class(mask):
            sub_pop_fitness = jnp.where(mask[:, jnp.newaxis], merged_fitness, jnp.inf)
            sub_indices = jnp.argsort(sub_pop_fitness[:, 0])[: self.pop_size]
            next_pop = merged_pop[sub_indices]
            next_fit = sub_pop_fitness[sub_indices]
            return next_pop, next_fit

        def handle_large_class(mask):
            # jax.debug.print("large_mask.sum: {}", jnp.sum(mask))
            sub_pop, sub_pop_fitness, _ = self.nsgaii_selection(
                state, merged_pop, merged_fitness, mask, class_num[i]
            )
            return sub_pop, sub_pop_fitness

        for i in range(self.k):
            mask = partition == i
            sub_pop, sub_pop_fitness = lax.cond(
                class_num[i] <= std_size,
                handle_small_class,
                handle_large_class,
                operand=mask,
            )

            all_sub_pop.append(sub_pop)
            all_sub_pop_fit.append(sub_pop_fitness)

        all_fit = jnp.vstack(all_sub_pop_fit)
        all_pop = jnp.vstack(all_sub_pop)

        indices = jnp.argsort(all_fit[:, 0])[: self.pop_size]
        next_generation = all_pop[indices]
        next_fitness = all_fit[indices]
        return state.update(population=next_generation, fitness=next_fitness)

    def get_class_num(self, state, partition):
        class_num = jnp.zeros((self.k,))
        std_size = jnp.ceil(self.pop_size / self.k).astype(jnp.int32)
        class_mask = jnp.zeros((self.k,))
        allocation = 0
        for i in range(self.k):
            mask = partition == i
            sub_pop_num = jnp.sum(mask).astype(jnp.int32)
            class_num = class_num.at[i].set(sub_pop_num)
            allocation += lax.cond(
                sub_pop_num <= std_size, lambda: sub_pop_num, lambda: std_size
            )
            class_mask = class_mask.at[i].set(
                lax.cond(sub_pop_num <= std_size, lambda: 0, lambda: 1)
            )
        all_remain = 2 * self.pop_size - allocation
        cur_remain = self.pop_size - allocation
        for i in range(self.k):
            num = lax.cond(
                class_mask[i],
                lambda: std_size
                + jnp.ceil((class_num[i] - std_size) * cur_remain / all_remain),
                lambda: class_num[i],
            )
            class_num = class_num.at[i].set(num)
        fin_remain = self.pop_size - jnp.sum(class_num)

        def loop_body(variables):
            remain, class_num = variables
            index = jnp.argmax(class_num)
            class_num = class_num.at[index].set(class_num[index] - 1)
            remain += 1
            return (remain, class_num)

        all_remain, class_num = lax.while_loop(
            lambda x: x[0] < 0, loop_body, (fin_remain, class_num)
        )
        return class_num.astype(jnp.int32), state

    def nsgaii_selection(self, state, merged_pop, merged_fitness, mask, sub_pop_size):
        """
        select certain number of sub_population by NSGA-II's selection algorithm.
        """
        fitness = jnp.where(mask[:, jnp.newaxis], merged_fitness, jnp.inf)
        population, fitness = non_dominate(
            population=merged_pop, fitness=fitness, topk=self.pop_size
        )
        mask_ = jnp.arange(self.pop_size) < sub_pop_size
        fitness = jnp.where(mask_[:, jnp.newaxis], fitness, jnp.inf)
        return population, fitness, state

    def gp_offspring_(self, state, population, mask, fitness):
        """
        get sub_population by Gaussian Process and mutation.

        Parameters
        ----------
            state: State
            population: N*D matrix
            mask: n*1 bool array
            fitness: N*M matrix
        """
        new_key, key, pop_key, x_key, mut_key = jax.random.split(state.key, 5)
        n = jnp.sum(mask)
        N = population.shape[0]
        D = population.shape[1]
        new_pop, new_fit = random_fill(
            matrix=population, matrix2=fitness, mask=mask, key=key
        )

        def normal_fun(x_key):
            _keys = random.split(x_key, num=3)

            # Determine the indices of the population that will be generated. Unselected individuals' indices are marked with -1 and their value will be jnp.nan.
            pop_indices = jnp.arange(N)
            final_pop_indices = jnp.where(
                pop_indices % jnp.floor(N / n) == 0, pop_indices, -1
            )

            # Randomly classify the dimensions of the population.
            d_indices = jnp.arange(D)
            shuffled_dim_indices = random.permutation(_keys[0], d_indices)
            arr = jnp.arange(self.n_objs)

            def get_dim_indices(i, shuffled_indices):
                return lax.dynamic_slice(shuffled_indices, (i * self.l,), (self.l,))

            shuffled_dim_indices_group = jax.vmap(
                lambda i: get_dim_indices(i, shuffled_dim_indices)
            )(arr)

            # Determine the lower and upper bounds of the population's fitness.
            fmin = 1.5 * jnp.nanmin(new_fit, axis=0) - 0.5 * jnp.nanmax(new_fit, axis=0)
            fmax = 1.5 * jnp.nanmax(new_fit, axis=0) - 0.5 * jnp.nanmin(new_fit, axis=0)

            # Train groups of GP models for each objective
            def gp_body(m, shuffled_indices_group, _keys, new_pop, new_fit):
                key_arr = random.split(_keys[m], num=3)

                # random select the group of population to train GP model.
                permutation = random.permutation(key_arr[0], pop_indices)
                parents = jnp.where(
                    pop_indices <= jnp.ceil(n / self.n_objs), permutation, -1
                )
                _mask = pop_indices == parents
                sub_off, sub_fit = random_fill(
                    matrix=new_pop, matrix2=new_fit, key=key_arr[1], mask=_mask
                )

                # select dimensions for each objective
                dim_indices = shuffled_indices_group[m]

                _keys = random.split(key_arr[2], num=self.l)
                likelihood = Gaussian(num_datapoints=len(sub_off))

                # get the prediction's input of the GP model
                inputs = jnp.linspace(fmin[m], fmax[m], N)
                inputs = jnp.where(final_pop_indices >= 0, inputs, jnp.inf)[
                    :, jnp.newaxis
                ]
                _sub_fit = lax.dynamic_slice(sub_fit, (0, m), (sub_fit.shape[0], 1))

                # get the offspring of the GP model
                def get_off(i, dim_indices, _keys, sub_off, sub_fit):
                    model = GPRegression(likelihood=likelihood, kernel=Linear())
                    _sub_pop = lax.dynamic_slice(
                        sub_off, (0, dim_indices[i]), (sub_fit.shape[0], 1)
                    )
                    model.fit(x=_sub_fit, y=_sub_pop, optimzer=ox.adam(0.001))
                    _, ymu, ystd = model.predict(inputs)
                    return ymu + random.normal(_keys[i], shape=ystd.shape) * ystd

                res = jax.vmap(
                    lambda i: get_off(i, dim_indices, _keys, sub_off, sub_fit)
                )(jnp.arange(self.l))
                return res

            x_keys = jax.random.split(x_key, num=self.n_objs)
            # sub_off is a list of offspring of each objective. M*L*N matrix
            sub_off = jax.vmap(
                lambda i: gp_body(
                    i, shuffled_dim_indices_group, x_keys, new_pop, new_fit
                )
            )(arr)

            # reshape sub_off: M*L*N matrix, to (M*L*N) matrix
            off = jnp.vstack(sub_off).T

            # replace the selected dimensions with the offspring.
            def get_offspring(i, offspring):
                index = shuffled_dim_indices[i]
                return offspring.at[:, index].set(off[:, i])

            offspring = lax.fori_loop(0, self.l * self.n_objs, get_offspring, new_pop)

            # Convert invalid values to random values
            rand_pop = jax.random.uniform(
                pop_key, offspring.shape, minval=self.lb, maxval=self.ub
            )
            invalid = (offspring < self.lb) | (offspring > self.ub)
            offspring = jnp.where(invalid, rand_pop, offspring)
            return offspring

        final_pop = lax.cond(
            n >= 2 * self.n_objs, normal_fun, lambda x: population, x_key
        )
        final_pop = self.mutation(mut_key, final_pop)
        return final_pop, state.update(key=new_key)
