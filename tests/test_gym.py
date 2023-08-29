from evox import pipelines, algorithms, problems
from evox.monitors import StdSOMonitor
from evox.utils import TreeAndVector
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


@pytest.mark.skip(reason="time consuming")
@pytest.mark.parametrize("batch_policy", [True, False])
def test_cartpole(batch_policy):
    key = jax.random.PRNGKey(42)
    model_key, pipeline_key = jax.random.split(key)

    model = CartpolePolicy()
    params = model.init(model_key, jnp.zeros((4,)))
    adapter = TreeAndVector(params)
    monitor = StdSOMonitor()
    problem = problems.rl.Gym(
        policy=jax.jit(model.apply),
        num_workers=4,
        env_per_worker=10,
        controller_options={
            "num_cpus": 0.25,
            "num_gpus": 0,
        },
        worker_options={"num_cpus": 1},
        batch_policy=batch_policy,
    )
    center = adapter.to_vector(params)
    # create a pipeline
    pipeline = pipelines.StdPipeline(
        algorithm=algorithms.PGPE(
            optimizer="adam",
            center_init=center,
            pop_size=40,
        ),
        problem=problem,
        pop_transform=adapter.batched_to_tree,
        fitness_transform=monitor.record_fit,
    )
    # init the pipeline
    state = pipeline.init(pipeline_key)

    # run the pipeline for 10 steps
    for i in range(10):
        print(monitor.get_min_fitness())
        state = pipeline.step(state)

    # the result should be close to 0
    min_fitness = monitor.get_min_fitness()
    print(min_fitness)
    # gym should be deterministic
    assert min_fitness == -83.0
