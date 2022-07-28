from .module import *


class Algorithm(Module):
    """Base class for all algorithms
    """
    def setup(self, key):
        return {}

    def tell(self, state, X, F):
        pass

    def ask(self, state):
        pass
