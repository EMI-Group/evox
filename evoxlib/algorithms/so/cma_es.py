import math
import jax
import jax.numpy as jnp
from jax import lax

import evoxlib as exl


@exl.jit_class
class CMA_ES(exl.Algorithm):
    def __init__(self, init_mean, init_var, pop_size=None, recombination_weights=None, c1=None, cμ=None, cc=None, cσ=None, dσ=None):
        """
        See [link](https://arxiv.org/pdf/1604.00772.pdf) for default argument values
        """
        assert(init_var > 0)
        N = init_mean.shape[0]
        if pop_size is None:
            M = 4 + math.floor(3 * math.log(self.dim))
        else:
            M = pop_size

        if recombination_weights is not None:
            w = recombination_weights
            assert(w.shape[0] == M)
            μeff = jnp.sum(w[:M//2]) ** 2 / jnp.sum(w[:M//2] ** 2)
        else:
            w = math.log((M + 1) / 2) - jnp.log(jnp.arange(1, M + 1))
            μeff = jnp.sum(w[:M//2]) ** 2 / jnp.sum(w[:M//2] ** 2)
        if c1 is None:
            c1 = 2 / ((N + 1.3) ** 2 + μeff)
        if cμ is None:
            cμ = min(1 - c1, 2 * ((μeff - 2 + 1 / μeff) / ((N + 2) ** 2 + μeff)))
        if cc is None:
            cc = (4 + μeff / N) / (N + 4 + 2 * μeff / N)
        if cσ is None:
            cσ = (2 + μeff) / (N + μeff + 5)
        if dσ is None:
            dσ = 1 + 2 * max(0, math.sqrt((μeff - 1) / (N + 1)) - 1) + cσ

        if recombination_weights is None:
            μeff_neg = jnp.sum(w[M//2:]) ** 2 / jnp.sum(w[M//2:] ** 2)
            αμ = 1 + c1 / cμ
            αμeff = 1 + 2 * μeff_neg / (μeff + 2)
            αpd = (1 - c1 - cμ) / N / cμ
            ω_possum = 1 / jnp.sum(w[: M//2])
            ω_negsum = -1 / jnp.sum(w[M//2:])
            w = jnp.where(w >= 0, ω_possum * w, ω_negsum * min(αμ, αμeff, αpd) * w)

        self.dim = N
        self.pop_size = M
        self.init_mean = init_mean
        self.init_var = init_var
        self.weight = w
        self.μeff = μeff
        self.positive_count = jnp.count_nonzero(w > 0)
        self.chiN = math.sqrt(self.dim) * (1 - 1 / 4 / self.dim + 1 / 21 / self.dim ** 2)
        self.c1 = c1
        self.cμ = cμ
        self.cc = cc
        self.cσ = cσ
        self.dσ = dσ

    def setup(self, key):
        state_key, init_key = jax.random.split(key)
        population = jax.random.normal(init_key, shape=(self.pop_size, self.dim))
        population = lax.map(lambda p, μ, σ: μ + σ * p, zip(population, self.init_mean, self.init_var))
        covariance = jnp.eye(self.dim)
        eigvecs = jnp.eye(self.dim)
        eigvals = jnp.ones((self.dim,))
        path_σ = jnp.zeros((self.dim,))
        path_c = jnp.zeros((self.dim,))
        mean = self.init_mean
        step_size = self.init_var

        return exl.State(population=population, zero_mean_pop=population,
                         covariance=covariance, eigvecs=eigvecs, eigvals=eigvals,
                         path_σ=path_σ, path_c=path_c,
                         mean=mean, step_size=step_size,
                         iter_count=0, key=state_key)

    def ask(self, state):
        return state, state.population

    def tell(self, state, fitness):
        C = state.covariance
        B = state.eigvecs
        D = state.eigvals
        population = state.population
        zero_mean_pop = state.zero_mean_pop
        mean = state.mean
        path_σ = state.path_σ
        path_c = state.path_c
        step_size = state.step_size
        iter_count=state.iter_count

        fitness, population, zero_mean_pop = lax.sort((fitness, population, zero_mean_pop), is_stable=False)
        mean = jnp.sum(lax.map(lambda w, x: w * x, zip(self.weight[:self.positive_count], population[:self.positive_count])), axis=1)
        zmean = jnp.sum(lax.map(lambda w, x: w * x, zip(self.weight[:self.positive_count], zero_mean_pop[:self.positive_count])), axis=1)

        path_σ = (1 - self.cσ) * path_σ + math.sqrt(self.cσ * (2 - self.cσ) * self.μeff) * (B @ zmean)
        hσ = jnp.linalg.norm(path_σ) / math.sqrt(1 - (1 - self.cσ) ** (2 * iter_count * self.pop_size / self.positive_count)) / self.chiN < (1.4 + 2/ (self.dim + 1))
        path_c = (1 - self.cc) * path_c + hσ * math.sqrt(self.cc * (2 - self.cc) * self.μeff) * (B @ (D * zmean))
        step_size = step_size * math.exp(self.cσ / self.dσ * (jnp.linalg.norm(path_σ) / self.chiN - 1))

        C = (1 - self.c1 - self.cμ) * C \
            + self.c1 * (path_c.reshape(self.dim, 1) @ path_c.reshape(1, self.dim) + ((1 - hσ) * self.cc * (2 - self.cc)) * C) \
            + self.cμ * (B @ (D * zero_mean_pop[:self.positive_count]) * self.weight) @ jnp.transpose(B @ (D * zero_mean_pop[:self.positive_count]))

        def updateBD():
            D, B = jnp.linalg.eigh(C, symmetrize_input=True)
            D = jnp.sqrt(D)
            return B, D
        
        def noUpdate():
            return B, D

        B, D = lax.cond(iter_count % math.ceil(1 / (self.c1 + self.cμ) / self.dim / 10) == 0, updateBD, noUpdate)

        key, sample_key = jax.random.split(state.key)
        zero_mean_pop = jax.random.normal(sample_key, shape=(self.pop_size, self.dim))
        population = lax.map(lambda p: B @ (D * p), zero_mean_pop)
        population = lax.map(lambda p, μ, σ: μ + σ * p, zip(population, mean, step_size))

        return state.update(population=population, zero_mean_pop=zero_mean_pop,
                            covariance=C, eigvecs=B, eigvals=D,
                            path_σ=path_σ, path_c=path_c,
                            mean=mean, step_size=step_size,
                            iter_count=iter_count + 1, key=key)
