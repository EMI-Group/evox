import evoxlib as exl
from evoxlib import algorithms, problems, pipelines
from evoxlib.problems.neuroevolution.models import SimpleCNN
import jax
import jax.numpy as jnp
import pytest
import time


class PartialPGPE(exl.algorithms.PGPE):
    def __init__(self, center_init):
        super().__init__(300, center_init, 'adam',
                         center_learning_rate=0.01, stdev_init=0.01)


def test_neuroevolution_treemap():
    start = time.perf_counter()
    # create a pipeline
    problem = exl.problems.neuroevolution.MNIST("./", 128, SimpleCNN())
    center_init = jax.tree_util.tree_map(
        lambda x: x.reshape(-1),
        problem.initial_params,
    )
    pipeline = pipelines.StdPipeline(
        algorithm=exl.algorithms.TreeAlgorithm(
            PartialPGPE, problem.initial_params, center_init),
        problem=problem,
        fitness_monitor=True
    )
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
    problem = problems.neuroevolution.MNIST(
        "./", 128, SimpleCNN())
    adapter = exl.utils.TreeToVector(problem.initial_params)
    algorithm = algorithms.PGPE(
        300,
        adapter.to_vector(problem.initial_params),
        'adam',
        center_learning_rate=0.01,
        stdev_init=0.01
    )
    pipeline = pipelines.AdapterPipeline(
        algorithm=algorithm,
        problem=problem,
        adapter=adapter,
        fitness_monitor=True
    )
    # init the pipeline
    key = jax.random.PRNGKey(42)
    state = pipeline.init(key)

    # run the pipeline for 100 steps
    for i in range(300):
        state = pipeline.step(state)

    # the result should be close to 0
    state, min_fitness = pipeline.get_min_fitness(state)
    print(f'Adapter loss: {min_fitness}  time: {time.perf_counter() - start}')
