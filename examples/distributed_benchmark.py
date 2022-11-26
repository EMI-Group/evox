import time

import evox
import jax
import jax.numpy as jnp
import pytest
from evox.utils import rank_based_fitness, compose
from evox import algorithms, pipelines, problems
from evox.monitors import FitnessMonitor
from flax import linen as nn


class SimpleCNN(nn.Module):
    """A simple CNN model."""

    @nn.compact
    def __call__(self, x):
        # normalize
        x = x / 255.0
        x = (x - 0.5) / 0.5

        x = nn.Conv(features=32, kernel_size=(3, 3), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(features=32, kernel_size=(3, 3), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(features=32, kernel_size=(3, 3), padding="VALID")(x)
        x = nn.relu(x)

        x = x.reshape(x.shape[0], -1)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x


def init_problem_and_model(key):
    model = SimpleCNN()
    batch_size = 256
    initial_params = jax.jit(model.init)(key, jnp.zeros((batch_size, 32, 32, 3)))
    problem = evox.problems.neuroevolution.TorchvisionDataset(
        root="./datasets",  # the path to cifar10 dataset
        batch_size=batch_size,
        forward_func=jax.jit(model.apply),
        dataset_name="cifar10",
        in_memory=True,
    )
    return initial_params, problem


def run_benchmark(num_gpus):
    key = jax.random.PRNGKey(42)
    pipeline_key, model_init_key = jax.random.split(key)
    initial_params, problem = init_problem_and_model(model_init_key)

    adapter = evox.utils.TreeAndVector(initial_params)
    monitor = FitnessMonitor()

    algorithm = algorithms.PGPE(
        150,
        adapter.to_vector(initial_params),
        "clipup",
        center_learning_rate=0.01,
        stdev_init=0.1,
        stdev_learning_rate=0.2
    )
    pipeline = pipelines.DistributedPipeline(
        algorithm=algorithm,
        problem=problem,
        pop_size=150,
        num_workers=num_gpus,
        options={"num_gpus": 1, "num_cpus": 8},
        pop_transform=adapter.batched_to_tree,
        global_fitness_transform=monitor.update
    )
    # init the pipeline
    state = pipeline.init(key)
    # warm up
    for i in range(5):
        pipeline.step(state)
        _state, accuracy = pipeline.valid(state, metric="accuracy")

    start = time.perf_counter()
    valid_acc = []
    # run the pipeline for 1000 steps
    for i in range(1000):
        state = pipeline.step(state)
        if (i + 1) % 100 == 0:
            state, accuracy = pipeline.valid(state, metric="accuracy")
            valid_acc.append(accuracy)

    return time.perf_counter() - start, valid_acc, monitor.history


MAX_GPUS = 6
result = {}
for i in range(1, MAX_GPUS + 1):
    print(f"using {i} gpus")
    runtime, valid_acc, train_loss = run_benchmark(i)
    print([jnp.max(x).item() for x in valid_acc])
    print(runtime)
    result[i] = {
        "runtime": runtime,
        "valid_acc": valid_acc,
        "train_loss": train_loss
    }

with open("exp/distributed_result.json", "w") as f:
    json.dump(result, f)
