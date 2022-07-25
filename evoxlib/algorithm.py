from .module import *

class Algorithm(Module):
    def setup(self):
        return {}

    @lift
    def tell(self, state, X, F):
        pass

    @lift
    def ask(self, state):
        pass