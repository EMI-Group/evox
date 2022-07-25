from .module import *

class Problem(Module):
    def setup(self):
        return {}

    @lift
    def evaluate(self, state, X):
        pass