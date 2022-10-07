import evoxlib as exl
from evoxlib import pipelines, algorithms, problems
from evoxlib.monitors import FitnessMonitor
from evoxlib.utils import TreeAndVector
import jax
import jax.numpy as jnp
from flax import linen as nn
import pytest


class CartpolePolicy(nn.Module):
    """A simple model for cartpole"""

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(32)(x)
        x = nn.sigmoid(x)
        x = nn.Dense(2)(x)
        return jnp.argmax(x)


class PongPolicy(nn.Module):
    """A simple model for cartpole"""

    @nn.compact
    def __call__(self, x):
        # input is (210, 160, 3)
        x = x / 255.0
        x = nn.Conv(16, (3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (2, 2))
        x = nn.Conv(32, (3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (2, 2))
        x = nn.Conv(32, (3, 3))(x)
        x = nn.relu(x)
        x = nn.Conv(32, (3, 3))(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(6)(x)

        return jnp.argmax(x)


def test_cartpole():
    key = jax.random.PRNGKey(42)
    model_key, pipeline_key = jax.random.split(key)

    model = CartpolePolicy()
    params = model.init(model_key, jnp.zeros((4,)))
    adapter = TreeAndVector(params)
    monitor = FitnessMonitor()
    problem = problems.neuroevolution.Gym(
        policy=jax.jit(model.apply),
        num_workers=10,
        env_per_worker=20,
        controller_options={
            "num_cpus": 0,
            "num_gpus": 1,
        },
        worker_options={"num_cpus": 1},
    )
    center = adapter.to_vector(params)
    lb = center - jnp.abs(center) * 0.7
    ub = center + jnp.abs(center) * 0.7
    # create a pipeline
    pipeline = pipelines.StdPipeline(
        algorithm=algorithms.PGPE(
            optimizer="adam",
            center_init=center,
            # ub=ub,
            # lb=lb,
            # phi=0,
            pop_size=200,
        ),
        problem=problem,
        pop_transform=adapter.batched_to_tree,
        fitness_transform=monitor.update,
    )
    # init the pipeline
    state = pipeline.init(pipeline_key)

    # run the pipeline for 100 steps
    for i in range(100):
        print(monitor.get_min_fitness())
        state = pipeline.step(state)

    state, sample_pop = pipeline.sample(state)
    problem._render(adapter.to_tree(sample_pop[0]))
    # the result should be close to 0
    min_fitness = monitor.get_min_fitness()
    print(min_fitness)
    print(monitor.history)
    # assert min_fitness < 1e-4
    # pipeline.health_check(state)

def test_pong():
    key = jax.random.PRNGKey(42)
    model_key, pipeline_key = jax.random.split(key)

    model = PongPolicy()
    params = model.init(model_key, jnp.zeros((210, 160, 3)))
    adapter = TreeAndVector(params)
    monitor = FitnessMonitor()
    problem = problems.neuroevolution.Gym(
        env_name="ALE/Pong-v5",
        policy=jax.jit(model.apply),
        num_workers=10,
        env_per_worker=20,
        controller_options={
            "num_cpus": 0,
            "num_gpus": 1,
        },
        worker_options={"num_cpus": 1},
    )
    center = adapter.to_vector(params)
    # create a pipeline
    pipeline = pipelines.StdPipeline(
        algorithm=algorithms.PGPE(
            optimizer="adam",
            center_init=center,
            pop_size=200,
        ),
        problem=problem,
        pop_transform=adapter.batched_to_tree,
        fitness_transform=monitor.update,
    )
    # init the pipeline
    state = pipeline.init(pipeline_key)

    # run the pipeline for 100 steps
    for i in range(100):
        print(monitor.get_min_fitness())
        state = pipeline.step(state)

    state, sample_pop = pipeline.sample(state)
    problem._render(adapter.to_tree(sample_pop[0]))
    # the result should be close to 0
    min_fitness = monitor.get_min_fitness()
    print(min_fitness)
    print(monitor.history)
