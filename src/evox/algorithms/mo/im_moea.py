# Q. Zhang, A. Zhou, and Y. Jin, RM-MEDA: A regularity model-based multiobjective estimation of distribution algorithm,
# IEEE Transactions on Evolutionary Computation, 2008, 12(1): 41-63.


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
from evox.operators.gaussian_process.regression import GPRegression
import gpjax as gpx
from gpjax.kernels import Linear
from gpjax.likelihoods import Gaussian
from gpjax.mean_functions import Zero
import optax as ox
import sys
import os
import jax.lax as lax
from jax.experimental.host_callback import id_print

def random_fill(key, matrix, matrix2, mask):
    # indices = jnp.arange(matrix.shape[0])
    # masked_indices = jnp.where(mask, indices, jnp.inf)
    # sorted_indices = jnp.sort(masked_indices)
    # max_indices = jnp.sum(mask)
    # random_array = random.randint(key=key, shape=(matrix.shape[0],), minval=0, maxval=max_indices)
    # new_indices = sorted_indices[random_array].astype(jnp.int32)
    # matrix = matrix[new_indices, :]
    # matrix2 = matrix2[new_indices, :]

    matrix = jnp.where(mask, matrix, jnp.nan)
    matrix2 = jnp.where(mask, matrix2, jnp.nan)
    return matrix, matrix2

class IMMOEA(Algorithm):
    def __init__(
        self,
        lb,
        ub,
        n_objs=3,
        pop_size=100,
        l=3,
        k=10,
        selection_op=None,
        mutation_op=None,
    ):
        # make sure din > l * n_objs
        self.lb = lb
        self.ub = ub
        self.n_objs = n_objs
        self.dim = lb.shape[0]
        self.pop_size = pop_size
        self.l = l
        self.k = k
        self.selection = selection_op
        self.mutation = mutation_op

        if self.selection is None:
            self.selection = selection.UniformRand(1)
        if self.mutation is None:
            self.mutation = mutation.Polynomial((self.lb, self.ub))

    # initial set
    def setup(self, key):
        key, subkey = jax.random.split(key)
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
        )

    # generate initial population
    def init_ask(self, state):
        return state.population, state

    def init_tell(self, state, fitness):
        state = state.update(fitness=fitness)
        return state

    # generate next generation
    def ask(self, state):
        # Block the output of gpjax
        # old_stdout = sys.stdout
        # new_stdout = open(os.devnull, "w")
        # sys.stdout = new_stdout
        population = state.population
        fitness = state.fitness
        K = self.k
        W = jax.random.uniform(state.key, (K, self.n_objs))
        W = jnp.fliplr(jnp.sort(jnp.fliplr(W), axis=1))
        self.pop_size = (jnp.ceil(self.pop_size / K) * K).astype(jnp.int32)
        distances = cos_dist(fitness, W)
        partition = jnp.argmax(distances, axis=1)
        sub_pops = []
        for i in range(K):
            mask = partition == i
            # sub_pop = population[mask]
            # sub_fit = fitness[mask]
            # sub_pops.append(self._gen_offspring(state, sub_pop, sub_fit))
            sub_pops.append(self._gp_offspring_(state, population, mask, fitness))
        OffspringDec = jnp.vstack(sub_pops)
        print(OffspringDec.shape)
        # sys.stdout = old_stdout
        # new_stdout.close()
        return OffspringDec, state.update(
            next_generation=OffspringDec, partition=partition
        )

    # select next generation
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

    def _gen_offspring(self, state, sub_pop, sub_fit):
        key, sel_key, x_key, mut_key = jax.random.split(state.key, 4)
        N, D = sub_pop.shape
        L = 3
        Offspring = sub_pop
        if len(sub_pop) >= 2 * self.n_objs:
            indices = jnp.arange(D)
            shuffled_indices = random.permutation(x_key, indices)
            while len(shuffled_indices) < L * self.n_objs:
                _key, x_key = random.split(x_key)
                shuffled_indices = jnp.vstack(
                    (shuffled_indices, random.permutation(x_key, indices))
                )
            start = 0
            fmin = 1.5 * jnp.min(sub_fit, axis=0) - 0.5 * jnp.max(sub_fit, axis=0)
            fmax = 1.5 * jnp.max(sub_fit, axis=0) - 0.5 * jnp.min(sub_fit, axis=0)
            Offspring = None
            sel_keys = jax.random.split(sel_key, num=self.n_objs)
            x_keys = jax.random.split(x_key, num=L * self.n_objs)
            # Train one groups of GP models for each objective
            for m in range(self.n_objs):
                permutation = random.permutation(sel_keys[m], N)
                parents = permutation[: jnp.floor(N / self.n_objs).astype(jnp.int32)]
                sub_off = sub_pop[parents, :]
                indices = shuffled_indices[start : start + L]
                keys = x_keys[start : start + L]
                start += L
                likelihood = Gaussian(num_datapoints=len(parents))
                model = GPRegression(likelihood=likelihood, kernel=Linear())
                for d, key in zip(indices, keys):
                    model.fit_scipy(
                        x=sub_fit[parents, m : m + 1], y=sub_off[:, d : d + 1]
                    )
                    inputs = jnp.linspace(fmin[m], fmax[m], sub_off.shape[0])[
                        :, jnp.newaxis
                    ]
                    ymu, ystd = model.predict(inputs)
                    # get next generation
                    sub_off = sub_off.at[:, d].set(
                        ymu + random.normal(key, shape=ystd.shape) * ystd
                    )
                if Offspring is None:
                    Offspring = sub_off
                else:
                    Offspring = jnp.vstack((Offspring, sub_off))

        # Convert invalid values to random values
        randDec = jax.random.uniform(
            x_key, Offspring.shape, minval=self.lb, maxval=self.ub
        )
        invalid = (Offspring < self.lb) | (Offspring > self.ub)
        Offspring = jnp.where(invalid, randDec, Offspring)

        Offspring = self.mutation(mut_key, Offspring)
        next_generation = jnp.clip(Offspring, self.lb, self.ub)
        return next_generation

    def _gp_offspring_(self, state, population, mask, fitness):
        key, sel_key, x_key, mut_key = jax.random.split(state.key, 4)
        n = jnp.sum(mask)
        N = population.shape[0]
        D = population.shape[1]
        L = 3
        Offspring = population
        new_pop, new_fit = random_fill(matrix=population, matrix2=fitness, mask=mask, key=key)
        print("new_pop shape:{}".format(new_pop.shape))

        # if N >= 2 * self.n_objs:
        def normal_fun(x_key):
            d_indices = jnp.arange(D)
            num_indices = jnp.arange(N)
            shuffled_indices = random.permutation(x_key, d_indices)
            shuffled_indices_group = jnp.zeros(shape=(self.n_objs,L),dtype=jnp.int32)
            arr = jnp.array(jnp.arange(self.n_objs+1))
            # print("arr shape:{}".format(arr.shape))
            def get_dim_indices(i, shuffled_indices):
            # for i in range(self.n_objs):
                return lax.dynamic_slice(shuffled_indices, (i*L,), (L,))
            shuffled_indices_group = jax.vmap(lambda i: get_dim_indices(i, shuffled_indices))(arr)
            # start = 0
            fmin = 1.5 * jnp.nanmin(new_fit, axis=0) - 0.5 * jnp.nanmax(new_fit, axis=0)
            fmax = 1.5 * jnp.nanmax(new_fit, axis=0) - 0.5 * jnp.nanmin(new_fit, axis=0)
            Offspring = None
            # sel_keys = jax.random.split(sel_key, num=self.n_objs)
            x_keys = jax.random.split(x_key, num=L * self.n_objs)
            # Train one groups of GP models for each objective
            def gp_body(m, shuffled_indices_group, x_keys, new_pop, new_fit):
            # for m in range(self.n_objs):
                #
                _keys = random.split(x_keys[m], num=3)
                permutation = random.permutation(_keys[0], num_indices)
                parents = jnp.where(num_indices, permutation, -1)
                # parents = jnp.where((jnp.arange(N) < group_num), permutation, N)
                # sub_off = jnp.where((jnp.arange(N) == parents)[:, jnp.newaxis], new_pop, jnp.nan)
                # sub_fit = jnp.where((jnp.arange(N) == parents)[:, jnp.newaxis], new_fit, jnp.nan)
                _mask = jnp.arange(new_pop.shape[0]) == parents
                sub_off, sub_fit = random_fill(matrix=new_pop, matrix2=new_fit, key=_keys[1], mask=_mask)
                # sub_off = sorted_pop[parents, :]
                # dim_indices = shuffled_indices[start : start + L]
                # keys = x_keys[start : start + L]
                # start += L

                dim_indices = shuffled_indices_group[m]
                keys = random.split(_keys[2], num=L)
                likelihood = Gaussian(num_datapoints=len(sub_off))
                model = GPRegression(likelihood=likelihood, kernel=Linear())
                res = None
                # for d, key in zip(dim_indices, keys):
                def get_off(i, dim_indices, keys, model, sub_off, sub_fit):
                    d, key = dim_indices[i], keys[i]
                    _sub_fit = lax.dynamic_slice(sub_fit, (0,m), (sub_fit.shape[0], 1))
                    _sub_off = lax.dynamic_slice(sub_off, (0,d), (sub_fit.shape[0], 1))
                    # model.fit_scipy(
                    #     x=_sub_fit, y=_sub_off
                    # )
                    model.fit( x=_sub_fit, y=_sub_off, optimzer=ox.sgd(0.001))
                    inputs = jnp.linspace(fmin[m], fmax[m], len(parents))[
                        :, jnp.newaxis
                    ]
                    _, ymu, ystd = model.predict(inputs)
                    # get next generation
                    # new_pop = sub_off.at[:, d].set(
                    #     ymu + random.normal(key, shape=ystd.shape) * ystd
                    # )
                    rrr = ymu + random.normal(key, shape=ystd.shape) * ystd
                    print("rrr shape:{}".format(rrr.shape))
                    return ymu + random.normal(key, shape=ystd.shape) * ystd

                res = jax.vmap(lambda i: get_off(i, dim_indices, keys, model, sub_off, sub_fit), out_axes=1)(jnp.arange(L))
                print("res shape:{}".format(res.shape))

                # Offspring = jax.cond(Offspring is None, lambda x: x, lambda x: jnp.vstack((Offspring, x)), sub_off)
                # if Offspring is None:
                #     Offspring = sub_off
                # else:
                #     Offspring = jnp.vstack((Offspring, sub_off))

                return res


            sub_off = jax.vmap(lambda i : gp_body(i, shuffled_indices_group, x_keys, new_pop, new_fit))(jnp.arange(self.n_objs))
            print("shuffled_off:{}".format(sub_off.shape))
            off = jnp.zeros(shape=(new_pop.shape[0],self.n_objs*L))
            for i in range(sub_off.shape[1]):
                off = off.at[i].set(sub_off[:,i,:].reshape(off.shape[1]))

            # bool_index = jnp.zeros(new_pop.shape, dtype=bool)
            # bool_index = bool_index.at[:, :L * self.n_objs].set(True)
            offspring = jnp.zeros_like(new_pop,dtype=jnp.float32)
            d_num = L * self.n_objs
            for i in range(d_num):
                offspring = offspring.at[:,i].set(off[:,i])
            for i in range(D-d_num):
                offspring = offspring.at[:, i+d_num].set(new_pop[:, i+d_num])
            # shuffled_off = jnp.where(bool_index, off, new_pop)
            # index = jnp.argsort(shuffled_indices)
            # offspring = shuffled_off[index]
            return offspring
        Offspring = lax.cond(n >= 2 * self.n_objs, normal_fun, lambda x : Offspring, x_key)

        # Convert invalid values to random values
        randDec = jax.random.uniform(
            x_key, Offspring.shape, minval=self.lb, maxval=self.ub
        )
        invalid = (Offspring < self.lb) | (Offspring > self.ub)
        Offspring = jnp.where(invalid, randDec, Offspring)

        Offspring = self.mutation(mut_key, Offspring)
        next_generation = jnp.clip(Offspring, self.lb, self.ub)
        return next_generation


