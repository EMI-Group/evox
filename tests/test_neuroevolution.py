import evoxlib as exl
from evoxlib.problems.neuroevolution.models import SimpleCNN
import jax
import jax.numpy as jnp
import pytest


class PartialCSO(exl.algorithms.CSO):
    def __init__(self, lb, ub):
        super().__init__(lb, ub, pop_size=20)


class Pipeline(exl.Module):
    def __init__(self):
        # MNIST with LeNet
        self.problem = exl.problems.neuroevolution.MNIST("./", 128, SimpleCNN())
        # [-1, 1] is used for lower and upper bound
        lb = jax.tree_util.tree_map(
            lambda x: -jnp.ones_like(x).reshape(-1),
            self.problem.initial_params,
        )
        ub = jax.tree_util.tree_map(
            lambda x: jnp.ones_like(x).reshape(-1),
            self.problem.initial_params,
        )

        # CSO
        self.algorithm = exl.algorithms.TreeAlgorithm(PartialCSO, self.problem.initial_params, lb, ub)

    def setup(self, key):
        # record the min fitness
        return exl.State({"min_fitness": 1e9})

    def step(self, state):
        # one step
        state, X = self.algorithm.ask(state)
        state, F = self.problem.evaluate(state, X)
        state = self.algorithm.tell(state, X, F)
        print(state.min_fitness)
        return state | {"min_fitness": jnp.minimum(state["min_fitness"], jnp.min(F))}

    def get_min_fitness(self, state):
        return state, state["min_fitness"]


def test_neuroevolution():
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
    print(min_fitness)
