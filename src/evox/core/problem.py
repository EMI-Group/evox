from typing import Tuple, Union, Any

import jax

from .module import *
from .state import State


class Problem(Stateful):
    """Base class for all problems"""

    def evaluate(
        self, state: State, pop: Union[jax.Array, Any]
    ) -> Tuple[jax.Array, State]:
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
            The new state of the problem.
        ndarray
            The fitness.
        """
        return jnp.empty(0)
