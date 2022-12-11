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

obv_shape = 7
act_shape = 4

class AgentModel(nn.Module):
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(32)(x)
        x = nn.tanh(x)
        x = nn.Dense(32)(x)
        x = nn.tanh(x)
        x = nn.Dense(act_shape)(x)
        return x

class AgentPolicy():
    def __init__(self, model, act_noise_std=0.01):
        self.model = model
        self.act_noise_std = act_noise_std

    def act_(self, params, obv):
        obv = obv / 100 - 1
        return self.model.apply(params, obv)

    def epsilon_greedy(self, params, obv, epsilon=0.1, seed=None):
        out = self.act_(params, obv)
        a = jnp.argmax(out)
        random_action = jax.random.randint(seed, shape=(), minval=0, maxval=4)
        action = jnp.where(jax.random.uniform(seed) < epsilon, random_action, a)
        return action

    def choice_with_p(self, params, obv, seed=None):
        out = self.act_(params, obv)

        # noise = jax.random.normal(seed, shape=(4, )) * self.act_noise_std
        # out += noise
        out = nn.softmax(out)

        a = jax.random.choice(seed, a=jnp.arange(act_shape), p=out)
        return a
    
    def choice(self, params, obv, seed=None):
        out = self.act_(params, obv)
        return jnp.argmax(out)

env_args = {"maze_idx": args.maze_idx, "time_limit": 200}

def main():
    batch_policy = False
    key = jax.random.PRNGKey(42)
    model_key, pipeline_key = jax.random.split(key)

    model = AgentModel()
    params = model.init(model_key, jnp.zeros((obv_shape,)))
    adapter = TreeAndVector(params)

    policy = AgentPolicy(model)

    monitor = GymMonitor(need_save=True, save_interval=10, env_name=env_name, monitor_info=args.maze_idx)
    problem = problems.neuroevolution.Gym(
        policy=jax.jit(policy.choice_with_p),
        num_workers=16,
        env_per_worker=25,
        controller_options={
            "num_cpus": 1,
            "num_gpus": 1,
        },
        worker_options={"num_cpus": 1, "num_gpus": 1/16},
        batch_policy=batch_policy,
        env_name=env_name,
        env_options = env_args,
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
        # algorithm=algorithms.CSO(
        #     lb=jnp.full((len(center),), -1),
        #     ub=jnp.full((len(center),), 1),
        #     pop_size=400,
        # ),
        # algorithm=algorithms.OpenES(
        #     init_params=center,
        #     pop_size=400,
        #     learning_rate=0.02,
        #     noise_std=0.02,
        # ),
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