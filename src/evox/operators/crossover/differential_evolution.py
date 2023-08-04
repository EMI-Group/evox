import jax.numpy as jnp
from jax import jit, random, vmap
from evox import jit_class


def _random_scaling(key, x):
    batch, dim = x.shape
    candidates = jnp.tile(x, (3, 1))  # shape: (3*batch, dim)
    candidates = random.permutation(key, candidates, axis=0)
    return candidates.reshape(batch, 3, dim)


def _de_mutation(parents, F):
    # use DE/rand/1
    # parents[0], parents[1] and parents[2] may be the same with each other, or with the original individual
    mutated_individual = parents[0] + F * (parents[1] - parents[2])
    return mutated_individual


def _de_crossover(key, new_x, x, CR):
    batch, dim = x.shape
    random_crossover = random.uniform(key, shape=(batch, dim))
    mask = random_crossover <= CR
    return jnp.where(mask, new_x, x)


@jit
def differential_evolve(key, x, F, CR):
    scaling_key, crossover_key = random.split(key, 2)
    scaled = _random_scaling(scaling_key, x)
    mutated_individual = vmap(_de_mutation)(scaled, F)
    children = _de_crossover(crossover_key, mutated_individual, x, CR)
    return children


@jit_class
class DifferentialEvolve:
    def __init__(self, F=0.5, CR=0.7):
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

    def __call__(self, key, x):
        return differential_evolve(key, x, self.F, self.CR)
