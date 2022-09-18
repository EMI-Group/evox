import evoxlib as exl

import jax
import jax.numpy as jnp


@exl.jit_class
class TournamentSelection(exl.Operator):
    def __init__(self, p):
        self.p = p

    def setup(self, key):
        return exl.State(key=key)

    def __call__(self, state, x, fitness):
        key, subkey = jax.random.split(state.key)
        num = int(x.shape[0] * self.p)
        chosen = jax.random.choice(subkey, x.shape[0], shape=(num,))
        candidates = x[chosen, :]
        best = max(candidates, key=lambda x: fitness(x))
        return exl.State(key=key), best

