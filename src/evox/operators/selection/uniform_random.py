<<<<<<< HEAD
import jax
import jax.numpy as jnp
from evox import jit_class, Operator, State


@jit_class
class UniformRandomSelection(Operator):
    def __init__(self, p):
        self.p = p

    def setup(self, key):
        return State(key=key)

    def __call__(self, state, x):
        key, subkey = jax.random.split(state.key)
        num = int(x.shape[0] * self.p)
        chosen = jax.random.choice(subkey, x.shape[0], shape=(num,))
        return chosen, State(key=key)
=======
from jax import random, jit
from evox import jit_class

from functools import partial


@partial(jit, static_argnames="prob")
def uniform_rand(key, pop, *others, prob):
    num = int(pop.shape[0] * prob)
    chosen = random.choice(key, pop.shape[0], shape=(num,))
    if len(others) == 0:
        return pop[chosen, :]
    else:
        return (pop[chosen, :], *[other[chosen] for other in others])


@jit_class
class UniformRand:
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, key, pop, *others):
        return uniform_rand(key, pop, *others, prob=self.prob)
>>>>>>> e9d1a7a9a7ff3bb82fc0c14c4cd4180929c822b9
