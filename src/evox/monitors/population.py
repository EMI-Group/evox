import jax.numpy as jnp
import numpy as np
import warnings


def get_init_range(data):
    """Given a numpy array, return a tuple (float, float) used for ploting"""

    min_val = np.min(data)
    max_val = np.max(data)
    range = max_val - min_val
    padding = range * 0.1
    return (min_val - padding, max_val + padding)


class PopulationMonitor:
    def __init__(self, n):
        warnings.warn(
            "PopulationMonitor is deprecated, please use StdSOMonitor",
            DeprecationWarning,
        )
        # single object for now
        assert n < 3
        self.n = n
        self.history = []
        self.min_fitness = float("inf")

    def update(self, pop):
        # convert to numpy array to save gpu memory
        self.history.append(np.array(pop).T)
