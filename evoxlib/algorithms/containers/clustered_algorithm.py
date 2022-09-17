import copy
from functools import partial

import chex
import jax
import jax.numpy as jnp
from evoxlib import Algorithm, State


class ClusterdAlgorithm(Algorithm):
    """A container that split the encoding into subproblems, and run an Algorithm on each.

    Can take in any base algorithm, split the problem into n different sub-problems
    and solve each problem using the base algorithm. Dim must be a multiple of num_cluster.
    """

    def __init__(self, base_algorithm, dim, num_cluster, *args, **kargs):
        assert dim % num_cluster == 0
        self.dim = dim
        self.num_cluster = num_cluster
        self.subproblem_dim = self.dim // num_cluster
        self._base_algorithm = base_algorithm(*args, **kargs)

    def init(self, key: chex.PRNGKey = None, name: str = "_top_level"):
        self.name = name
        keys = jax.random.split(key, self.num_cluster)
        self_state = self.setup(key)
        child_states = {
            "_base_algorithm": jax.vmap(
                partial(self._base_algorithm.init, name="_base_algorithm")
            )(keys)
        }
        return self_state._set_child_states(child_states)

    def ask(self, state: State):
        state, xs = jax.vmap(self._base_algorithm.ask)(state)
        # concatenate different parts as a whole
        xs = jnp.concatenate(xs, axis=1)
        return state, xs

    def tell(self, state: State, fitness: chex.Array):
        def partial_tell(state):
            return self._base_algorithm.tell(state, fitness)
        return jax.vmap(partial_tell)(state)


class RandomMaskAlgorithm(Algorithm):
    pass
