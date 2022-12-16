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

envs = {"Ant-v4": {"observation_shape": (27), "action_shape": (8)}, 
        "HalfCheetah-v4": {"observation_shape": (17), "action_shape": (6)},
        "Hopper-v4": {"observation_shape": (11), "action_shape": (3)},
        "Humanoid-v4": {"observation_shape": (376), "action_shape": (17), "lb": -0.4, "ub": 0.4},
        "HumanoidStandup-v4": {"observation_shape": (376), "action_shape": (17), "lb": -0.4, "ub": 0.4},
        "InvertedDoublePendulum-v4": {"observation_shape": (11), "action_shape": (1)},
        "InvertedPendulum-v4": {"observation_shape": (4), "action_shape": (1)},
        "Reacher-v4": {"observation_shape": (11), "action_shape": (2)},
        "Swimmer-v4": {"observation_shape": (8), "action_shape": (2)},
        "Walker2d-v4": {"observation_shape": (17), "action_shape": (6)}}

train_args = {
    "num_workers": 16,
    "env_per_worker": 25,
    "act_noise_std": 0.01,
}

openai_es_args = {

    "learning_rate": 0.01,
    "noise_std": 0.02,
    "optimizer": 'adam',
    "mirrored_sampling": True,
    "utility": True,
    "l2coeff": 0.005,
}

argsparser = argparse.ArgumentParser()
argsparser.add_argument("--env", type=str, default="Ant-v4")
# argsparser.add_argument("--need_save", type=bool, default=400)

args = argsparser.parse_args()
env_name = args.env
obv_shape = envs[env_name]["observation_shape"]
act_shape = envs[env_name]["action_shape"]

class AgentModel(nn.Module):
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(32, kernel_init=jax.nn.initializers.xavier_uniform())(x)
        x = nn.tanh(x)
        x = nn.Dense(32, kernel_init=jax.nn.initializers.xavier_uniform())(x)
        x = nn.tanh(x)
        x = nn.Dense(act_shape, kernel_init=jax.nn.initializers.xavier_uniform())(x)
        x = nn.tanh(x)  # action is in range(-1, 1)
        
        return x

class AgentPolicy():
    def __init__(self, model, act_noise_std=0.01):
        self.model = model
        self.act_noise_std = act_noise_std
    
    def act_(self, params, obv):
        return self.model.apply(params, obv)

    def act_with_noise(self, params, obv, seed):
        action = self.model.apply(params, obv)
        noise = jax.random.normal(seed, shape=action.shape) * self.act_noise_std
        action += noise

        if "lb" in envs[env_name] and "ub" in envs[env_name]:
            action = jnp.clip(action, envs[env_name]["lb"], envs[env_name]["ub"])
        # print(f"action: {action}, noise: {noise}")
        return action


def main():
    batch_policy = False
    key = jax.random.PRNGKey(41)
    model_key, pipeline_key = jax.random.split(key)

    model = AgentModel()
    params = model.init(model_key, jnp.zeros((obv_shape,)))
    adapter = TreeAndVector(params)

    policy = AgentPolicy(model, act_noise_std=train_args["act_noise_std"])

    monitor = GymMonitor(need_save=True, save_interval=10000, env_name=env_name)
    problem = problems.neuroevolution.Gym(
        policy=jax.jit(policy.act_with_noise),
        num_workers=train_args["num_workers"],
        env_per_worker=train_args["env_per_worker"],
        controller_options={
            "num_cpus": 1,
            "num_gpus": 1,
        },
        worker_options={"num_cpus": 1, "num_gpus": 1/train_args["num_workers"]},
        batch_policy=batch_policy,
        env_name=env_name,
        normalize_obv=True
    )

    center = adapter.to_vector(params)

    print(f"the length of gene is {len(center)}")
    # create a pipeline
    pipeline = pipelines.GymPipeline(
        # algorithm=algorithms.PGPE(
        #     optimizer="adam",
        #     center_init=center,
        #     pop_size=train_args["num_workers"] * train_args["env_per_worker"],
        # ),
        # algorithm=algorithms.CSO(
        #     lb=jnp.full((len(center),), -2),
        #     ub=jnp.full((len(center),), 2),
        #     pop_size=train_args["num_workers"] * train_args["env_per_worker"],
        # ),
        # algorithm=algorithms.CSO_new(
        #     lb=jnp.full((len(center),), -2),
        #     ub=jnp.full((len(center),), 2),
        #     pop_size=train_args["num_workers"] * train_args["env_per_worker"],
        #     elite_rate=0.5
        # ),
        # algorithm=algorithms.OpenES(
        #     init_params=center,
        #     pop_size=train_args["num_workers"] * train_args["env_per_worker"],
        #     **openai_es_args
        # ),
        algorithm=algorithms.CMA_ES(
            init_mean=center,
            init_stdvar=0.1,
            pop_size=train_args["num_workers"] * train_args["env_per_worker"],
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