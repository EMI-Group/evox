import jax.numpy as jnp
import warnings


class FitnessMonitor:
    def __init__(self, n_objects=1, keep_global_best=True):
        # single object for now
        # assert n_objects == 1
        warnings.warn(
            "FitnessMonitor is deprecated, please use StdSOMonitor",
            DeprecationWarning,
        )
        self.n_objects = n_objects
        self.history = []
        self.min_fitness = float("inf")
        self.keep_global_best = keep_global_best

    def update(self, fitness):
        if self.n_objects > 1:
            self.history.append(fitness)
        else:
            if self.keep_global_best:
                self.min_fitness = min(self.min_fitness, jnp.min(fitness).item())
            else:
                self.min_fitness = jnp.min(fitness).item()
            self.history.append(self.min_fitness)
        return fitness

    def get_min_fitness(self):
        return self.min_fitness

    def get_history(self):
        return self.history

    def get_last(self):
        return self.history[-1]
