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

# 定义一个函数来计算后验均值和方差
def inf_exact(posterior, X, y, test_inputs):
    Kff = posterior.kernel(X, X) + posterior.likelihood.jitter * jnp.eye(X.shape[0])
    Kfs = posterior.kernel(X, test_inputs)
    Kss = posterior.kernel(test_inputs, test_inputs)
    Lf = jnp.linalg.cholesky(Kff)
    A = jnp.linalg.solve(Lf, Kfs)
    mu = Zero(test_inputs) + A.T @ jnp.linalg.solve(Lf.T, y - Zero(X))
    v = jnp.linalg.solve(Lf, y - Zero(X))
    cov = Kss - A.T @ A + jnp.outer(v, v)
    return mu, cov

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
        population = state.population
        fitness = state.fitness
        K = self.k
        W = jax.random.uniform(state.key, (K, self.n_objs))
        W = jnp.fliplr(jnp.sort(jnp.fliplr(W), axis=1))
        self.pop_size = jnp.ceil(self.pop_size / K) * K
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
        key, sel_key1, x_key, mut_key = jax.random.split(state.key, 4)
        N, D = sub_pop.shape
        L = 3
        OffDec = sub_pop
        # Gaussian process based reproduction
        # if len(sub_pop) < 2 * self.n_objs:
        #     OffDec = sub_pop
        if len(sub_pop) >= 2 * self.n_objs:
            fmin = 1.5 * jnp.min(sub_fit, axis=0) - 0.5 * jnp.max(sub_fit, axis=0)
            fmax = 1.5 * jnp.max(sub_fit, axis=0) - 0.5 * jnp.min(sub_fit, axis=0)
            # Train one groups of GP models for each objective
            for m in range(self.n_objs):
                permutation = jax.random.permutation(sel_key1, N)
                parents = permutation[:jnp.floor(N / self.n_objs).astype(jnp.int32)]
                offDec = sub_pop[parents,:]
                permutation = jax.random.permutation(x_key, D)
                for d in permutation[:L]:
                    # 定义输入数据
                    inputs = jnp.linspace(fmin[m], fmax[m], offDec.shape[0])
                    likelihood = Gaussian(num_datapoints=len(parents))
                    model = GPRegression(likelihood=likelihood)
                    model.fit(x=offDec, y=sub_fit[parents,:])
                    ymu, ys2 = model.predict(inputs)
                    # 生成后代的决策
                    offDec = offDec.at[:,d].set( ymu + jax.random.normal(x_key, shape=ys2.shape) * jnp.sqrt(ys2))
                OffDec = jnp.vstack((OffDec, offDec))

        # Convert invalid values to random values
        # Lower = jnp.tile(self.lb, (N, D))
        # Upper = jnp.tile(self.ub, (N, D))
        randDec = jax.random.uniform(x_key, OffDec.shape, minval=self.lb, maxval=self.ub)
        invalid = (OffDec < self.lb) | (OffDec > self.ub)
        OffDec = jnp.where(invalid, randDec, sub_pop)

        OffDec = self.mutation(mut_key, OffDec)
        next_generation = jnp.clip(OffDec, self.lb, self.ub)
        return next_generation
