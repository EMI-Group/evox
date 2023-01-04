import evox as ex
import jax
import jax.numpy as jnp


def _random_scaling(key, x):
    batch, dim = x.shape
    candidates = jnp.tile(x, (3, 1))  # shape: (3*batch, dim)
    candidates = jax.random.permutation(key, candidates, axis=0)
    return candidates.reshape(batch, 3, dim)


def _de_mutation(parents, F):
    # use DE/rand/1
    # parents[0], parents[1] and parents[2] may be the same with each other, or with the original individual
    mutated_individual = parents[0] + F * (parents[1] - parents[2])
    return mutated_individual


def _de_crossover(key, new_x, x, cr):
    batch, dim = x.shape
    random_crossover = jax.random.uniform(key, shape=(batch, dim))
    mask = random_crossover <= cr
    return jnp.where(mask, new_x, x)


@ex.jit_class
class DECrossover(ex.Operator):
    def __init__(self, stdvar=1.0, F=0.5, cr=0.7):
        """
        Parameters
        ----------
        F
            The scaling factor
        cr
            The probability of crossover
        """
        self.stdvar = stdvar
        self.F = F
        self.cr = cr

    def setup(self, key):
        return ex.State(key=key)

    def __call__(self, state, x):
        key = state.key
        key, scaling_key, crossover_key = jax.random.split(key, 3)
        scaled = _random_scaling(scaling_key, x)
        mutated_individual = jax.vmap(_de_mutation)(scaled, self.F)
        children = _de_crossover(crossover_key, mutated_individual, x, self.cr)
        return ex.State(key=key), children
