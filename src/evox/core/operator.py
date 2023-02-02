from typing import Tuple

import jax.numpy as jnp

from .module import *
from .state import State


class Operator(Stateful):
    def __call__(self, state: State, pop: jnp.ndarray) -> Tuple[jnp.ndarray, State]:
        return jnp.empty(0), State()
