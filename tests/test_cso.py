import evoxlib as exl
import jax
import jax.numpy as jnp
import pytest


@exl.jit_class
@exl.use_state_class
class Pipeline(exl.Module):
    def __init__(self):
        # choose an algorithm
        self.algorithm = exl.algorithms.CSO(
            lb=jnp.full(shape=(2,), fill_value=-32),
            ub=jnp.full(shape=(2,), fill_value=32),
            pop_size=100,
        )
        # choose a problem
        self.problem = exl.problems.classic.Ackley()

    def setup(self, key):
        # record the min fitness
        return {"min_fitness": 1e9}

    def step(self, state):
        # one step
        state, X = self.algorithm.ask(state)
        state, F = self.problem.evaluate(state, X)
        state = self.algorithm.tell(state, X, F)
        return state | {"min_fitness": jnp.minimum(state["min_fitness"], jnp.min(F))}

    def get_min_fitness(self, state):
        return state, state["min_fitness"]


def test_cso():
    # create a pipeline
    pipeline = Pipeline()
    # init the pipeline
    key = jax.random.PRNGKey(42)
    state = pipeline.init(key)

    # run the pipeline for 100 steps
    for i in range(100):
        state = pipeline.step(state)

    # the result should be close to 0
    state, min_fitness = pipeline.get_min_fitness(state)
    assert min_fitness < 1e-4
