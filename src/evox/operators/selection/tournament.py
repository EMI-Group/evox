from typing import Callable

import jax.numpy as jnp
from jax import random, jit, vmap
from evox import jit_class

from functools import partial


@partial(jit, static_argnums=[3, 4, 5])
def tournament_single_fit(key, pop, fit, n_round, tournament_func, tournament_size):
    chosen = random.choice(key, n_round, shape=(n_round, tournament_size))
    candidates_fitness = fit[chosen, ...]
    winner_indices = vmap(tournament_func)(candidates_fitness)
    index = chosen[jnp.arange(n_round), winner_indices]
    return pop[index], index


@partial(jit, static_argnums=[3, 4, 5])
def tournament_multi_fit(key, pop, fit, n_round, tournament_func, tournament_size):
    chosen = random.choice(key, n_round, shape=(n_round, tournament_size))
    candidates_fitness = fit[chosen, ...]
    winner_indices = vmap(jnp.lexsort)(jnp.transpose(candidates_fitness, (0, 2, 1)))
    index = chosen[jnp.arange(n_round), winner_indices[:, 0]]
    return pop[index], index


@jit_class
class Tournament:
    """Tournament selection"""

    def __init__(
        self,
        n_round: int,
        tournament_func: Callable = jnp.argmin,
        tournament_size: int = 2,
        multi_objective: bool = False
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
        self.multi_obj = multi_objective

    def __call__(self, key, pop, *args):
        if self.multi_obj:
            fit = jnp.c_[args]
            return tournament_multi_fit(
                key, pop, fit, self.n_round, self.tournament_func, self.tournament_size
            )
        else:
            fit = args[0]
            return tournament_single_fit(
                key, pop, fit, self.n_round, self.tournament_func, self.tournament_size
            )
