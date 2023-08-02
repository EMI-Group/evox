from typing import Callable

import jax.numpy as jnp
from jax import random, jit, vmap
from evox import jit_class

from functools import partial


@partial(jit, static_argnums=[3, 4, 5])
def tournament(key, pop, fit, n_round, tournament_func, tournament_size):
    # select num_round times and each time
    # k individuals to form candidates
    chosen = random.choice(key, n_round, shape=(n_round, tournament_size))
    # candidates = x[chosen, ...]
    candidates_fitness = fit[chosen, ...]
    winner_indices = vmap(tournament_func)(candidates_fitness)
    index = jnp.diagonal(chosen[:, winner_indices])
    return pop[index]


@jit_class
class Tournament:
    """Tournament selection"""

    def __init__(
        self,
        n_round: int,
        tournament_func: Callable = jnp.argmax,
        tournament_size: int = 2,
    ):
        """
        Parameters
        ----------
        num_round
            Number of time the tournament will hold.
        tournament_func
            A function used to determine the winner of the tournament.
            The function must accept a array of shape (N, ) or (N, K),
            where N is the number of individuals and K is the number of objectives,
            and return a single integer indicating the index of the winner.
        tournament_size
            Number of individuals in one tournament
        """
        self.n_round = n_round
        self.tournament_func = tournament_func
        self.tournament_size = tournament_size

    def __call__(self, key, pop, fit):
        return tournament(
            key, pop, fit, self.n_round, self.tournament_func, self.tournament_size
        )
