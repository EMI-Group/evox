import types
from typing import Tuple

import jax

from .module import *
from .state import State


class Algorithm(Stateful):
    """Base class for all algorithms"""

    def ask(self, state: State) -> Tuple[jax.Array, State]:
        """Ask the algorithm

        Ask the algorithm for points to explore

        Parameters
        ----------
        state
            The state of this algorithm.

        Returns
        -------
        population
            The candidate solution.
        state
            The new state of the algorithm.
        """
        return jnp.zeros(0), State()

    def tell(self, state: State, fitness: jax.Array) -> State:
        """Tell the algorithm more information

        Tell the algorithm about the points it chose and their corresponding fitness

        Parameters
        ----------
        state
            The state of this algorithm
        fitness
            The fitness

        Returns
        -------
        state
            The new state of the algorithm
        """
        return State()


def has_init_ask(algorithm):
    return hasattr(algorithm, "init_ask") and callable(algorithm.init_ask)


def has_init_tell(algorithm):
    return hasattr(algorithm, "init_tell") and callable(algorithm.init_tell)
