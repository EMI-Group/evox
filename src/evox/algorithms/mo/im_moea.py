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
from evox.operators.gaussian_processes.regression import GPRegression
import gpjax as gpx
from gpjax.kernels import Linear
from gpjax.likelihoods import Gaussian
from gpjax.mean_functions import Zero
import optax as ox
import logging

class IMMOEA(Algorithm):
    def __init__(
        self,
        lb,
        ub,
        n_objs=3,
        pop_size=100,
        l=3,
        k = 10,
        selection_op=None,
        mutation_op=None,
    ):
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
        logging.basicConfig(level=logging.CRITICAL)
        population = state.population
        fitness = state.fitness
        K = self.k
        W = jax.random.uniform(state.key, (K, self.n_objs))
        W = jnp.fliplr(jnp.sort(jnp.fliplr(W), axis=1))
        self.pop_size = int(jnp.ceil(self.pop_size / K) * K)
        distances = cos_dist(fitness, W)
        partition = jnp.argmax(distances, axis=1)
        sub_pops = []
        for i in range(K):
            mask = partition == i
            sub_pop = population[mask]
            sub_fit = fitness[mask]
            sub_pops.append(self._gen_offspring(state, sub_pop, sub_fit))
        OffspringDec = jnp.vstack(sub_pops)
        return OffspringDec, state.update(next_generation=OffspringDec, partition=partition)

    # select next generation
    def tell(self, state, fitness):
        merged_pop = jnp.concatenate([state.population, state.next_generation], axis=0)
        merged_fitness = jnp.concatenate([state.fitness, fitness], axis=0)

        rank = non_dominated_sort(merged_fitness)
        order = jnp.argsort(rank)
        worst_rank = rank[order[self.pop_size]]
        mask = rank == worst_rank
        last = jnp.sum(mask)
        next = self.pop_size - last
        while last > self.pop_size - next:
            crowding_dis = crowding_distance(merged_fitness, mask)
            order = jnp.argsort(crowding_dis)
            mask[order[-1]] = False
            last -= 1

        survivor = merged_pop[mask]
        survivor_fitness = merged_fitness[mask]
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
                shuffled_indices = jnp.vstack((shuffled_indices, random.permutation(x_key, indices)))
            start = 0
            fmin = 1.5 * jnp.min(sub_fit, axis=0) - 0.5 * jnp.max(sub_fit, axis=0)
            fmax = 1.5 * jnp.max(sub_fit, axis=0) - 0.5 * jnp.min(sub_fit, axis=0)
            Offspring = None
            sel_keys = jax.random.split(sel_key, num=self.n_objs)
            x_keys = jax.random.split(x_key, num=L * self.n_objs)
            # Train one groups of GP models for each objective
            for m in range(self.n_objs):
                permutation = random.permutation(sel_keys[m], N)
                parents = permutation[:jnp.floor(N / self.n_objs).astype(jnp.int32)]
                sub_off = sub_pop[parents,:]
                indices = shuffled_indices[start:start+L]
                keys = x_keys[start:start+L]
                start += L
                likelihood = Gaussian(num_datapoints=len(parents))
                model = GPRegression(likelihood=likelihood)
                for d, key in zip(indices, keys):
                    model.fit_scipy(x=sub_fit[parents,m:m+1], y=sub_off[:,d:d+1])
                    inputs = jnp.linspace(fmin[m], fmax[m], sub_off.shape[0])
                    ymu, ystd = model.predict(inputs)
                    # get next generation
                    sub_off = sub_off.at[:,d].set(ymu + random.normal(key, shape=ystd.shape) * ystd)
                if Offspring is None:
                    Offspring = sub_off
                else:
                    Offspring = jnp.vstack((Offspring, sub_off))

        # Convert invalid values to random values
        randDec = jax.random.uniform(x_key, Offspring.shape, minval=self.lb, maxval=self.ub)
        invalid = (Offspring < self.lb) | (Offspring > self.ub)
        Offspring = jnp.where(invalid, randDec, Offspring)

        Offspring = self.mutation(mut_key, Offspring)
        next_generation = jnp.clip(Offspring, self.lb, self.ub)
        return next_generation
