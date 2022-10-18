import types

import chex

from .module import *
from .state import State


class Problem(Stateful):
    """Base class for all algorithms"""

    def evaluate(self, state: State, pop: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the fitness at given points

        Parameters
        ----------
        state : dict
            The state of this problem.
        X : ndarray
            The population.

        Returns
        -------
        dict
            The new state of the algorithm.
        ndarray
            The fitness.
        """
        pass
