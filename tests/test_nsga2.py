import evoxlib as exl
import jax
import jax.numpy as jnp
import pytest

@exl.use_state_class
class Pipeline(exl.Module):
    def __init__(self):
        # choose an algorithm
        self.algorithm = exl.algorithms.NSGA2(
            lb=jnp.full(shape=(2,), fill_value=0),
            ub=jnp.full(shape=(2,), fill_value=1),
            pop_size=100,
        )
        # choose a problem
        self.problem = exl.problems.classic.ZDT1(n=2)

    def step(self, state):
        # one step
        state, X = self.algorithm.ask(state)
        state, F = self.problem.evaluate(state, X)
        state = self.algorithm.tell(state, X, F)
        return state

def test_nsga2():
    key = jax.random.PRNGKey(123)
    pipeline = Pipeline()
    state = pipeline.init(key)

    for i in range(100):
        state = pipeline.step(state)