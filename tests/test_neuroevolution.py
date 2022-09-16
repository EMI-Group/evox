import evoxlib as exl
from evoxlib.problems.neuroevolution.models import SimpleCNN
import jax
import jax.numpy as jnp
import pytest
import time


class PartialPGPE(exl.algorithms.PGPE):
    def __init__(self, center_init):
        super().__init__(300, center_init, 'adam', center_learning_rate=0.01, stdev_init=0.01)


class TreemapPipeline(exl.Module):
    def __init__(self):
        # MNIST with LeNet
        self.problem = exl.problems.neuroevolution.MNIST("./", 128, SimpleCNN())
        center_init = jax.tree_util.tree_map(
            lambda x: x.reshape(-1),
            self.problem.initial_params,
        )
        self.algorithm = exl.algorithms.TreeAlgorithm(PartialPGPE, self.problem.initial_params, center_init)

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


class AdapterPipeline(exl.Module):
    def __init__(self):
        # MNIST with LeNet
        self.problem = exl.problems.neuroevolution.MNIST("./", 128, SimpleCNN())
        self.adapter = exl.utils.TreeToVector(self.problem.initial_params)
        self.algorithm = exl.algorithms.PGPE(
            300,
            self.adapter.to_vector(self.problem.initial_params),
            'adam',
            center_learning_rate=0.01,
            stdev_init=0.01
        )

    def setup(self, key):
        # record the min fitness
        return exl.State({"min_fitness": 1e9})

    def step(self, state):
        # one step
        state, X = self.algorithm.ask(state)
        tree_X = self.adapter.to_tree(X, True)
        state, F = self.problem.evaluate(state, tree_X)
        state = self.algorithm.tell(state, X, F)
        return state | {"min_fitness": jnp.minimum(state["min_fitness"], jnp.min(F))}

    def get_min_fitness(self, state):
        return state, state["min_fitness"]


def test_neuroevolution_treemap():
    start = time.perf_counter()
    # create a pipeline
    pipeline = TreemapPipeline()
    # init the pipeline
    key = jax.random.PRNGKey(42)
    state = pipeline.init(key)

    # run the pipeline for 100 steps
    for i in range(300):
        state = pipeline.step(state)

    # the result should be close to 0
    state, min_fitness = pipeline.get_min_fitness(state)
    print(f'Treemap loss: {min_fitness}  time: {time.perf_counter() - start}')


def test_neuroevolution_adapter():
    start = time.perf_counter()
    # create a pipeline
    pipeline = AdapterPipeline()
    # init the pipeline
    key = jax.random.PRNGKey(42)
    state = pipeline.init(key)

    # run the pipeline for 100 steps
    for i in range(300):
        state = pipeline.step(state)

    # the result should be close to 0
    state, min_fitness = pipeline.get_min_fitness(state)
    print(f'Adapter loss: {min_fitness}  time: {time.perf_counter() - start}')