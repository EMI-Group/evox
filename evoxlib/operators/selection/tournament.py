from turtle import pos
import evoxlib as exl

import jax
import jax.numpy as jnp


@exl.jit_class
class TournamentSelection(exl.Operator):
    def __init__(self, p, k):
        self.p = p
        self.k = k

    def setup(self, key):
        return exl.State(key=key)

    def __call__(self, state, x, fitness):
        key, subkey = jax.random.split(state.key)
        # randomly select k individuals to form candidates
        chosen = jax.random.choice(subkey, x.shape[0], shape=(self.k,))
        candidates = x[chosen,:]
        candidates_fitness = fitness[chosen]
        # sort candidates
        order = jnp.argsort(candidates_fitness)[::-1]
        candidates = candidates[order]

        possibilities = jnp.full((self.k, ), 1 - self.p)
        possibilities[0] = 1
        possibilities = jnp.cumprod(possibilities) * self.p

        # select an individual from k candidates by possibilities
        chosen = jax.random.choice(subkey, self.k, key=possibilities, shape=(1,))
        return exl.State(key=key), candidates[chosen, :]