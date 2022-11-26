import evox as ex
from evox import pipelines, algorithms, problems
from evox.monitors import GymMonitor
from evox.utils import TreeAndVector
import jax
import jax.numpy as jnp
from flax import linen as nn
import argparse

envs = {"HalfCheetah-v4": {"observation_shape": (17), "action_shape": (6), "mo_keys": ["reward_run", "reward_ctrl"]},}

argsparser = argparse.ArgumentParser()
argsparser.add_argument("--env", type=str, default="Ant-v4")
argsparser.add_argument("--pop_size", type=int, default=400)
# argsparser.add_argument("--need_save", type=bool, default=400)

args = argsparser.parse_args()
env_name = args.env
obv_shape = envs[env_name]["observation_shape"]
act_shape = envs[env_name]["action_shape"]

class AgentModel(nn.Module):
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        x = nn.Dense(act_shape)(x)
        x = nn.tanh(x)  # action is in range(-1, 1)
        return x

class AgentPolicy():
    def __init__(self, model, random_key, act_noise_std=0.01):
        self.model = model
        self.random_key = random_key
        self.act_noise_std = act_noise_std
    
    def act_(self, params, obv):
        return self.model.apply(params, obv)

    def act_with_noise(self, params, obv):
        self.random_key, _ = jax.random.split(self.random_key)
        action = self.model.apply(params, obv)
        noise = jax.random.normal(self.random_key, shape=action.shape) * self.act_noise_std
        # print(f"action: {action}, noise: {noise}")
        return action + noise

def main():
    batch_policy = False
    key = jax.random.PRNGKey(42)
    model_key, pipeline_key = jax.random.split(key)
    model_key, policy_key = jax.random.split(model_key)

    model = AgentModel()
    params = model.init(model_key, jnp.zeros((obv_shape,)))
    adapter = TreeAndVector(params)

    policy = AgentPolicy(model, policy_key)

    monitor = GymMonitor(need_save=True, save_interval=10, env_name=env_name, mo_keys=envs[env_name]["mo_keys"])
    problem = problems.neuroevolution.Gym(
        policy=jax.jit(policy.act_with_noise),
        num_workers=16,
        env_per_worker=25,
        controller_options={
            "num_cpus": 1,
            "num_gpus": 1,
        },
        worker_options={"num_cpus": 1, "num_gpus": 1/16},
        batch_policy=batch_policy,
        env_name=env_name,
        mo_keys=envs[env_name]["mo_keys"],
    )

    center = adapter.to_vector(params)
    print(f"the length of gene is {len(center)}")
    # create a pipeline
    pipeline = pipelines.GymPipeline(
        algorithm=ex.algorithms.NSGA2(
            lb=jnp.full(shape=(len(center),), fill_value=-1),
            ub=jnp.full(shape=(len(center),), fill_value=1),
            n_objs=len(envs[env_name]["mo_keys"]),
            pop_size=400,
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