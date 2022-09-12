import evoxlib as exl
import jax
import jax.numpy as jnp
import pytest


class Pipeline(exl.Module):
    def __init__(self):
        center = jax.random.uniform(jax.random.PRNGKey(0), shape=(2, ), minval=-5, maxval=5)
        # choose an algorithm
        self.algorithm = exl.algorithms.PGPE(
            pop_size=100,
            center_init=center,
            optimizer='adam',
            center_learning_rate=0.3,
            stdev_init=1,
            stdev_learning_rate=0.1
        )
        # choose a problem
        self.problem = exl.problems.classic.Rastrigin()

    def setup(self, key):
        # record the min fitness
        return exl.State({"min_fitness": 1e9})

    def step(self, state):
        # one step
        state, X = self.algorithm.ask(state)
        state, F = self.problem.evaluate(state, X)
        state = self.algorithm.tell(state, X, F)

        return state | {"min_fitness": jnp.minimum(state["min_fitness"], jnp.min(F))}

    def get_min_fitness(self, state):
        return state, state["min_fitness"]


def test_pgpe():
    # create a pipeline
    pipeline = Pipeline()
    # init the pipeline
    key = jax.random.PRNGKey(42)
    state = pipeline.init(key)

    # run the pipeline for 100 steps
    for i in range(1000):
        state = pipeline.step(state)
        state, min_fitness = pipeline.get_min_fitness(state)

    # the result should be close to 0
    state, min_fitness = pipeline.get_min_fitness(state)
    print(min_fitness)
    assert min_fitness < 1e-1
