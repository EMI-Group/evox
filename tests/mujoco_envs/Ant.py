import sys
sys.path.append('/home/lishuang/evox')

import evox as ex
from evox import pipelines, algorithms, problems
from evox.monitors import GymMonitor
from evox.utils import TreeAndVector
import jax
import jax.numpy as jnp
from flax import linen as nn


class AntPolicy(nn.Module):
    

    # for Ant-v4
    """
    observation: 27 [-inf, inf]
    action: Box(-1.0, 1.0, (8,), float32)
    """
    observation_shape = (27)

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(32)(x)
        x = nn.sigmoid(x)
        x = nn.Dense(32)(x)
        x = nn.sigmoid(x)
        x = nn.Dense(8)(x)
        x = nn.tanh(x)  # action is in range(-1, 1)
        return x


if __name__ == "__main__":
    batch_policy = False
    key = jax.random.PRNGKey(42)
    model_key, pipeline_key = jax.random.split(key)

    model = AntPolicy()
    params = model.init(model_key, jnp.zeros((27,)))
    adapter = TreeAndVector(params)

    monitor = GymMonitor(need_save=True, save_interval=100)
    problem = problems.neuroevolution.Gym(
        policy=jax.jit(model.apply),
        num_workers=16,
        env_per_worker=25,
        controller_options={
            "num_cpus": 1,
            "num_gpus": 0,
        },
        worker_options={"num_cpus": 1, "num_gpus": 1/16},
        batch_policy=batch_policy,
        env_name="Ant-v4",
        env_options={"ctrl_cost_weight": 0, "contact_cost_weight": 0}
    )
    center = adapter.to_vector(params)
    print(f"the length of gene is {len(center)}")
    # create a pipeline
    pipeline = pipelines.GymPipeline(
        algorithm=algorithms.PGPE(
            optimizer="adam",
            center_init=center,
            pop_size=400,
        ),
        problem=problem,
        pop_transform=adapter.batched_to_tree,
        monitor_update=monitor.update,
    )
    # init the pipeline
    state = pipeline.init(pipeline_key)

    # run the pipeline for 10 steps
    for i in range(100000):
        state = pipeline.step(state)
    # the result should be close to 0