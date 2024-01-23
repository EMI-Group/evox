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
        crossover_op=None,
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

    # generate next generation by mutation
    def ask(self, state):
        population = state.population
        fitness = state.fitness
        M = self.n_objs
        D = self.dim
        K = self.k
        W = jnp.random.rand(K, M)
        W = jnp.fliplr(jnp.sort(jnp.fliplr(W), axis=1))
        self.pop_size = jnp.ceil(self.pop_size / K) * K
        distances = cos_dist(population, W)
        partition = jnp.argmax(distances, axis=1)
        sub_pop = []
        for i in range(K):
            mask = partition == i
            sub_pop = population[mask]
            sub_fit = fitness[mask]
            sub_pop[i] = self._gen_offspring(state, sub_pop, sub_fit)

        OffspringDec = jnp.vstack(sub_pop)
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
        # Gaussian process based reproduction
        if len(sub_pop) < 2 * self.n_objs:
            OffDec = sub_pop
        else:
            fmin = 1.5 * jnp.min(sub_fit, axis=0) - 0.5 * jnp.max(sub_fit, axis=0)
            fmax = 1.5 * jnp.max(sub_fit, axis=0) - 0.5 * jnp.min(sub_fit, axis=0)
            # Train one groups of GP models for each objective
            # 定义高斯过程参数
            log_likelihood_noise = jnp.log(0.01)
            # 定义均值函数、协方差函数和似然函数
            mean_function = Zero()
            kernel = Linear()
            likelihood = Gaussian(log_likelihood_noise)
            # create gp model
            prior = gpx.gps.Prior(kernel=kernel, mean_function=mean_function)
            posterior = gpx.gps.NonConjugatePosterior(prior=prior, likelihood=likelihood)
            for m in range(self.n_objs):
                permutation = jax.random.permutation(sel_key1, N)
                parents = permutation[:jnp.floor(N / self.n_objs).astype(jnp.int32)]
                offDec = sub_pop[parents,:]
                permutation = jax.random.permutation(x_key, D)
                for d in permutation[:L]:
                    # 定义输入数据
                    inputs = jnp.linspace(fmin[m], fmax[m], offDec.shape[0])
                    model = gpx.GaussianProcess(kernel, mean_function, likelihood)
                    # 使用确切推理来获取后代的决策的均值和方差
                    # 这里需要你根据你的模型来定义infExact函数
                    ymu, ys2 = inf_exact(posterior, sub_pop[parents, m], sub_pop[parents, d], inputs)
                    # 生成后代的决策
                    offDec = offDec.at[:,d].set( ymu + jax.random.normal(x_key, shape=ys2.shape) * jnp.sqrt(ys2))
                OffDec = jnp.vstack((OffDec, offDec))

        # Convert invalid values to random values
        Lower = jnp.tile(self.lb, (N, D))
        Upper = jnp.tile(self.ub, (N, D))
        randDec = jax.vmap(jax.random.uniform)(x_key, Lower, Upper)
        invalid = (OffDec < Lower) | (OffDec > Upper)
        OffDec = jnp.where(invalid, randDec, sub_pop)

        OffDec = self.mutation(mut_key, OffDec)
        next_generation = jnp.clip(OffDec, self.lb, self.ub)
        return next_generation

    '''
         假设你已经有了父母的决策数据和其他必要的函数和数据结构
        # parents, m, d, PopObj, PopDec, fmin, fmax, offDec 都是预先定义的
        # 定义高斯过程参数
        log_likelihood_noise = jnp.log(0.01)
        # 定义均值函数、协方差函数和似然函数
        mean_function = Zero()
        kernel = Linear()
        likelihood = Gaussian(log_likelihood_noise)
        # 创建高斯过程模型
        prior = gpx.Prior(kernel=kernel, mean_function=mean_function)
        posterior = gpx.Posterior(prior=prior, likelihood=likelihood)
        # 定义输入数据
        inputs = jnp.linspace(fmin(m), fmax(m), offDec.shape[0])
        # 使用确切推理来获取后代的决策的均值和方差
        # 这里需要你根据你的模型来定义infExact函数
        ymu, ys2 = infExact(posterior, PopObj(parents, m), PopDec(parents, d), inputs)
        # 生成后代的决策
        offDec = ymu + jax.random.normal(random.PRNGKey(0), shape=ys2.shape) * jnp.sqrt(ys2)
        '''
