from .module import *


class Operator(Module):
    def __init_subclass__(cls):
        attr = getattr(cls, '__call__')
        assert isinstance(attr, types.FunctionType)
        wrapped = use_state(attr)
        setattr(cls, '__call__', wrapped)

    def __call__(self, state, x):
        pass