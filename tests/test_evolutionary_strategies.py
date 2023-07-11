import jax
from jax import jit, vmap, numpy as jnp
from flax import linen as nn

from src import evox
from src.evox import algorithms
from src.evox.problems.rl.gym_no_distribution import Gym
from src.evox.utils import TreeAndVector


class ClassicPolicy(nn.Module):
    """A simple model for Classic Control problem"""

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(16)(x)
        x = nn.relu(x)
        x = nn.Dense(2)(x)

        return jnp.argmax(x)


def cartpole(al: type, **kwargs):
    key = jax.random.PRNGKey(42)
    model_key, pipeline_key = jax.random.split(key)

    model = ClassicPolicy()
    params = model.init(model_key, jnp.zeros((4,)))
    adapter = TreeAndVector(params)
    center = adapter.to_vector(params)

    monitor = evox.monitors.StdSOMonitor()

    problem = Gym(
        policy=jit(vmap(model.apply)),
        env_name="CartPole-v1",
        env_options={"new_step_api": True},
        pop_size=64,
    )

    # create a pipeline
    pipeline = evox.pipelines.StdPipeline(
        algorithm=al(
            center_init=center,
            pop_size=64,
            **kwargs,
        ),
        problem=problem,
        pop_transform=adapter.batched_to_tree,
        fitness_transform=monitor.record_fit,
    )
    # init the pipeline
    state = pipeline.init(key)

    # run the pipeline for 10 steps
    for i in range(20):
        state = pipeline.step(state)
        # print(monitor.get_min_fitness())

    # obtain 500
    min_fitness = monitor.get_min_fitness()
    # cartpole is simple. expect to obtain max score(500) in each algorithm
    assert min_fitness == -500


def test_cma_es():
    cartpole(algorithms.CMAES, init_stdev=1)


def test_pgpe():
    cartpole(algorithms.PGPE, optimizer="adam")


def test_open_es():
    cartpole(algorithms.OpenES, learning_rate=0.02, noise_stdev=0.02, mirrored_sampling=True, optimizer="adam")


