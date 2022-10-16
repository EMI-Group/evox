from .module import *
from .state import State

class Operator(Stateful):
    def __call__(self, state: State, pop: jnp.ndarray) -> jnp.ndarray:
        pass
