from functools import partial

import jax
import jax.numpy as jnp
import copy

import evoxlib as exl


class ClusterdAlgorithm(exl.Algorithm):
    """A general clustered algorithm

    Can take in any base algorithm, split the problem into num_cluster different sub-problems
    and solve each problem using the base algorithm.
    """
    def __init__(self, lb, ub, algorithm, num_cluster, *args, **kargs):
        self.dim = lb.shape[0]
        assert self.dim % num_cluster == 0
        self.num_cluster = num_cluster
        self.subproblem_dim = self.dim // num_cluster
        self.lb = lb.reshape(num_cluster, self.subproblem_dim)
        self.ub = ub.reshape(num_cluster, self.subproblem_dim)

        # use the magic vmap_class to create num_cluster copies of the algorithm
        self.base_algorithm = exl.vmap_class(algorithm, num_cluster)(
            lb[: self.subproblem_dim], ub[: self.subproblem_dim], *args, **kargs
        )

    def ask(self, state):
        state, xs = self.base_algorithm.ask(state)
        # concatenate different parts as a whole
        xs = jnp.concatenate(xs, axis=1)
        return state, xs

    def tell(self, state, x, F):
        # split into different parts
        xs = jnp.transpose(
            x.T.reshape(self.num_cluster, -1, self.subproblem_dim), (0, 2, 1)
        )
        # copy fitness num_cluster times
        F = jnp.tile(F, (self.num_cluster, 1))
        return self.base_algorithm.tell(state, xs, F)
