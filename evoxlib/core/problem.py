import types

import chex

from .module import *


class Problem(Module):
    """Base class for all algorithms
    """
    def __init_subclass__(cls):
        for method_name in ['evaluate']:
            attr = getattr(cls, method_name)
            assert isinstance(attr, types.FunctionType)
            wrapped = use_state(attr)
            setattr(cls, method_name, wrapped)

    def setup(self, key: chex.PRNGKey = None):
        return {}

    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
        """Evaluate the fitness at given points

        Parameters
        ----------
        state : dict
            The state of this problem.
        X : ndarray
            The query points.

        Returns
        -------
        dict
            The new state of the algorithm.
        ndarray
            The fitness.
        """
        pass
