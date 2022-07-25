import chex

from .module import *

class Problem(Module):
    def setup(self, key: chex.PRNGKey = None):
        return {}

    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
        pass