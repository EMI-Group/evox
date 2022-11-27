import time

import evox
import evox as ex
import jax
import jax.numpy as jnp
import pytest
from evox import algorithms, pipelines, problems
from evox.monitors import FitnessMonitor
from flax import linen as nn


class SimpleCNN(nn.Module):
    """A simple CNN model."""

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=6, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(features=16, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=16, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(features=16, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = x.reshape(x.shape[0], -1)
        x = nn.Dense(120)(x)
        x = nn.sigmoid(x)
        x = nn.Dense(10)(x)
        return x


def init_problem_and_model(key):
    model = SimpleCNN()
    batch_size = 128
    initial_params = model.init(key, jnp.zeros((batch_size, 32, 32, 3)))
    problem = ex.problems.neuroevolution.TorchvisionDataset(
        root="./datasets",
        batch_size=batch_size,
        forward_func=model.apply,
        dataset_name="cifar10",
    )
    return initial_params, problem


def run_benchmark(num_gpus):
    key = jax.random.PRNGKey(42)
    pipeline_key, model_init_key = jax.random.split(key)
    initial_params, problem = init_problem_and_model(model_init_key)

    start = time.perf_counter()
    adapter = ex.utils.TreeAndVector(initial_params)
    monitor = FitnessMonitor()
    algorithm = algorithms.PGPE(
        100,
        adapter.to_vector(initial_params),
        "adam",
        center_learning_rate=0.01,
        stdev_init=0.01,
    )
    pipeline = pipelines.StdPipeline(
        algorithm=algorithm,
        problem=problem,
        pop_transform=adapter.batched_to_tree,
        fitness_transform=monitor.update,
    )
    # init the pipeline
    state = pipeline.init(key)

    # run the pipeline for 100 steps
    for i in range(3000):
        state = pipeline.step(state)

    # the result should be close to 0
    min_fitness = monitor.get_min_fitness()
    print(f"Adapter loss: {min_fitness}  time: {time.perf_counter() - start}")


MAX_GPUS = 1

for i in range(1, MAX_GPUS + 1):
    run_benchmark(i)
