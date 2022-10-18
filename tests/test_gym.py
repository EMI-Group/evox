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


# https://docs.ray.io/en/latest/ray-core/examples/plot_pong_example.html
def pong_preprocess(img):
    # Crop the image.
    img = img[35:195]
    # Downsample by factor of 2.
    img = img[::2, ::2, 0]
    # Erase background (background type 1 and 2).
    img = jnp.where((img == 144) | (img == 109), 0, img)
    # Set everything else (paddles, ball) to 1.
    img = jnp.where(img != 0, 1, img)
    return img


class PongPolicy(nn.Module):
    """A simple model for cartpole"""

    @nn.compact
    def __call__(self, img):
        x = pong_preprocess(img)
        x = x.astype(jnp.float32)
        x = x.reshape(-1)
        x = nn.Dense(128)(x)
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
    for i in range(20):
        print(monitor.get_min_fitness())
        state = pipeline.step(state)

    # the result should be close to 0
    min_fitness = monitor.get_min_fitness()
    print(min_fitness)
    print(monitor.history)
    assert min_fitness <= -200

def test_pong():
    key = jax.random.PRNGKey(42)
    model_key, pipeline_key = jax.random.split(key)

    model = PongPolicy()
    params = model.init(model_key, jnp.zeros((210, 160, 3)))
    adapter = TreeAndVector(params)
    monitor = FitnessMonitor()
    problem = problems.neuroevolution.Gym(
        env_name="ALE/Pong-v5",
        env_options={
            "full_action_space": False
        },
        policy=jax.jit(model.apply),
        num_workers=16,
        env_per_worker=4,
        controller_options={
            "num_cpus": 0,
            "num_gpus": 0,
        },
        worker_options={"num_cpus": 1, "num_gpus": 1/16},
        batch_policy=False
    )
    center = adapter.to_vector(params)
    # create a pipeline
    pipeline = pipelines.StdPipeline(
        algorithm=algorithms.PGPE(
            optimizer="adam",
            center_init=center,
            pop_size=64,
        ),
        problem=problem,
        pop_transform=adapter.batched_to_tree,
        fitness_transform=monitor.update,
    )
    # init the pipeline
    state = pipeline.init(pipeline_key)
    # run the pipeline for 100 steps
    for i in range(10):
        print(monitor.get_min_fitness())
        state = pipeline.step(state)

    state, sample_pop = pipeline.sample(state)
    # problem._render(adapter.to_tree(sample_pop[0]), ale_render_mode="human")
    # the result should be close to 0
    min_fitness = monitor.get_min_fitness()
    print(min_fitness)
    print(monitor.history)
