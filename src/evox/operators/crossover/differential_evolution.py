import jax.numpy as jnp
from jax import jit, random, vmap
from evox import jit_class


def _de_mutation(x1, x2, x3, F):
    mutated_pop = x1 + F * (x2 - x3)
    return mutated_pop


def _de_crossover(key, new_x, x, CR):
    batch, dim = x.shape
    random_crossover = random.uniform(key, shape=(batch, dim))
    mask = random_crossover < CR
    return jnp.where(mask, new_x, x)

@jit
def differential_evolve(key, x1, x2, x3, F, CR):
    key, de_key = random.split(key)
    mutated_pop = _de_mutation(x1, x2, x3, F)

    children = _de_crossover(de_key, mutated_pop, x1, CR)
    return children


@jit_class
class DifferentialEvolve:
    def __init__(self, F=0.5, CR=1):
        """
        Parameters
        ----------
        F
            The scaling factor
        CR
            The probability of crossover
        """
        self.F = F
        self.CR = CR

    def __call__(self, key, p1, p2, p3):
        return differential_evolve(key, p1, p2, p3, self.F, self.CR)
