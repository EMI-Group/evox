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
