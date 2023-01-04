import jax.numpy as jnp

import evox as ex


@ex.jit_class
class Sphere(ex.Problem):
    def __init__(self, a=20, b=0.2, c=2):
        self.a = a
        self.b = b
        self.c = c

    def evaluate(self, state, X):
        F = jnp.sum(X**2, axis=-1)
        return state, F
