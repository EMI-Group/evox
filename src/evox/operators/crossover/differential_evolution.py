import jax
import jax.numpy as jnp

from evox import jit_class, Operator, State
# def _random_scaling(key, x):
#     batch, dim = x.shape
#     candidates = jnp.tile(x, (3, 1))  # shape: (3*batch, dim)
#     candidates = jax.random.permutation(key, candidates, axis=0)
#     return candidates.reshape(batch, 3, dim)


def _de_mutation(x1, x2, x3, F):
    # use DE/rand/1
    # parents[0], parents[1] and parents[2] may be the same with each other, or with the original individual
    mutated_pop = x1 + F * (x2 - x3)
    return mutated_pop


def _de_crossover(key, new_x, x, cr):
    batch, dim = x.shape
    random_crossover = jax.random.uniform(key, shape=(batch, dim))
    mask = random_crossover < cr
    return jnp.where(mask, new_x, x)


@jit_class
class DECrossover(Operator):
    def __init__(self, F=0.5, cr=1):
        """
        Parameters
        ----------
        F
            The scaling factor
        cr
            The probability of crossover
        """
        self.F = F
        self.cr = cr

    def setup(self, key):
        return State(key=key)

    def __call__(self, state, *args):
        key = state.key
        x1, x2, x3 = args
        key, de_key = jax.random.split(key)
        mutated_pop = _de_mutation(x1, x2, x3, self.F)

        children = _de_crossover(de_key, mutated_pop, x1, self.cr)

        return children, State(key=key)
