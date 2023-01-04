from typing import Callable

import evox as ex
import jax
import jax.numpy as jnp


@ex.jit_class
class TournamentSelection(ex.Operator):
    """Tournament selection"""

    def __init__(
        self, num_round: int, tournament_func: Callable = jnp.argmax, tournament_size: int = 2
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
        self.num_round = num_round
        self.tournament_func = tournament_func
        self.tournament_size = tournament_size

    def setup(self, key):
        return ex.State(key=key)

    def __call__(self, state, x, fitness):
        key, subkey = jax.random.split(state.key)
        # select num_round times and each time
        # k individuals to form candidates
        chosen = jax.random.choice(subkey, self.num_round, shape=(
            self.num_round, self.tournament_size))
        # candidates = x[chosen, ...]
        candidates_fitness = fitness[chosen, ...]
        winner_indices = jax.vmap(self.tournament_func)(candidates_fitness)
        index = jnp.diagonal(chosen[:, winner_indices])
        return ex.State(key=key), x[index]
