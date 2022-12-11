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
import gym, time

obv_shape = 7
act_shape = 4
seed = 42
model_path = '/home/lishuang/evox/logs/Maze_navigation-v0/9/2022_12_05_02_20_53/best_iteration70.npy'
maze_idx = 9

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

def initialize_agent_policy():
    key = jax.random.PRNGKey(seed)
    model = AgentModel()
    initial_params = model.init(key, jnp.zeros(obv_shape)) 

    adapter = ex.utils.TreeAndVector(initial_params)
    policy_gene = jnp.load(model_path)
    # policy_params = adapter.to_tree(policy_gene)
    policy_params = adapter.to_tree(policy_gene)
    return model, policy_params

def main():
    model, params = initialize_agent_policy()
    policy = AgentPolicy(model)

    env = gym.make('Maze_navigation-v0', maze_idx=maze_idx, time_limit=200, easy_mode=False)
    observation, _ = env.reset(seed=42)
    total_reward = 0

    action_li = []
    while True:
        # action = policy.choice_with_p(params, observation)
        action = jax.jit(policy.choice)(params, observation, seed=None)
        action_li.append(int(action))
        
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        # print(observation, reward)
        if terminated or truncated:
            observation, info = env.reset()
            break

    env.close()
    print(action_li, total_reward)

if __name__ == '__main__':
    main()