import jax
from evox import jit_class, Operator, State


@jit_class
class GaussianMutation(Operator):
    def __init__(self, stdvar=1.0):
        self.stdvar = stdvar

    def setup(self, key):
        return State(key=key)

    def __call__(self, state, x):
        key, subkey = jax.random.split(state.key)
        perturbation = jax.random.normal(subkey, x.shape) * self.stdvar
        return x + perturbation, State(key=key)
