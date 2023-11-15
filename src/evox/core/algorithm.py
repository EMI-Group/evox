import types
from typing import Tuple

import jax

from .module import *
from .state import State


class Algorithm(Stateful):
    """Base class for all algorithms"""

    def init_ask(self, state: State) -> Tuple[jax.Array, State]:
        """Ask the algorithm for the initial population

        Override this method if you need to initialize the population in a special way.
        For example, Genetic Algorithm needs to evaluate the fitness of the initial population of size N,
        but after that, it only need to evaluate the fitness of the offspring of size M, and N != M.
        Since JAX requires the function return to have static shape, we need to have two different functions,
        one is the normal `ask` and another is `init_ask`.

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
        return None, State()

    def init_tell(self, state: State) -> State:
        """Tell the algorithm the fitness of the initial population
        Use in pair with `init_ask`.

        Parameters
        ----------
        state
            The state of this algorithm

        Returns
        -------
        state
            The new state of the algorithm
        """
        return State()

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
