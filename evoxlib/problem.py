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
        pass
