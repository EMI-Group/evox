import sys
sys.path.append('/home/lishuang/evox')

import evox as ex
from evox import pipelines, algorithms, problems
from evox.monitors import GymMonitor
from evox.utils import TreeAndVector
import jax
import jax.numpy as jnp
from flax import linen as nn
import argparse


argsparser = argparse.ArgumentParser()
argsparser.add_argument("--maze_idx", type=int, default=0)
argsparser.add_argument("--pop_size", type=int, default=400)
# argsparser.add_argument("--need_save", type=bool, default=400)

args = argsparser.parse_args()
env_name = "Maze_navigation-v0"

# in easy mode, the observation is the robot's position and face direction. The grid is 19 * 19.
obv_shape = 19 * 19 * 4 
act_shape = 4

class AgentModel(nn.Module):
    
    # table function
    @nn.compact
    def __call__(self, x):
        # x is in shape obs_shape
        x = nn.Dense(act_shape, use_bias=False)(x)
        return x

class AgentPolicy():
    def __init__(self, model):
        self.model = model
    
    def act(self, params, obv, seed=None):
        # obv is in shape obs_shape
        x, y, d = obv
        # assert x % 10 == 0 and y % 10 == 0 and 10 <= x <= 190 and 10 <= y <= 190

        x, y = x // 10 - 1, y // 10 - 1
        idx = x * 19 * 4 + y * 4 + d

        input_ = jnp.zeros((obv_shape, ))
        input_ = input_.at[jnp.int32(idx)].set(1)

        out = self.model.apply(params, input_)

        return jnp.argmax(out)
    
    def act_with_p(self, params, obv, seed=None):
    # obv is in shape obs_shape
        x, y, d = obv
        # assert x % 10 == 0 and y % 10 == 0 and 10 <= x <= 190 and 10 <= y <= 190

        x, y = x // 10 - 1, y // 10 - 1
        idx = x * 19 * 4 + y * 4 + d

        input_ = jnp.zeros((obv_shape, ))
        input_ = input_.at[jnp.int32(idx)].set(1)
        out = self.model.apply(params, input_)
        out = nn.softmax(out)
        a = jax.random.choice(seed, a=jnp.arange(act_shape), p=out)

        return a

def main():
    batch_policy = False
    key = jax.random.PRNGKey(42)
    model_key, pipeline_key = jax.random.split(key)

    model = AgentModel()
    params = model.init(model_key, jnp.zeros((obv_shape,)))
    adapter = TreeAndVector(params)

    policy = AgentPolicy(model)

    monitor = GymMonitor(need_save=False, save_interval=50, env_name=env_name)
    problem = problems.neuroevolution.Gym(
        policy=jax.jit(policy.act_with_p),
        num_workers=16,
        env_per_worker=25,
        controller_options={
            "num_cpus": 1,
            "num_gpus": 1,
        },
        worker_options={"num_cpus": 1, "num_gpus": 1/16},
        batch_policy=batch_policy,
        env_name=env_name,
        env_options = {"maze_idx": args.maze_idx, "time_limit": 300, "easy_mode": True},
    )

    center = adapter.to_vector(params)
    print(f"the length of gene is {len(center)}")
    # create a pipeline
    pipeline = pipelines.GymPipeline(
        # algorithm=algorithms.PGPE(
        #     optimizer="adam",
        #     center_init=center,
        #     pop_size=400,
        # ),
        # algorithm=algorithms.CSO(
        #     lb=jnp.full((len(center),), -1),
        #     ub=jnp.full((len(center),), 1),
        #     pop_size=400,
        # ),
        algorithm=algorithms.OpenES(
            init_params=center,
            pop_size=400,
            learning_rate=0.03,
            noise_std=0.03,
        ),
        problem=problem,
        pop_transform=adapter.batched_to_tree,
        monitor_update=monitor.update,
    )

    # init the pipeline
    state = pipeline.init(pipeline_key)

    # run the pipeline for 10 steps
    for _ in range(100000):
        state = pipeline.step(state)
    # the result should be close to 0


if __name__ == "__main__":
    main()