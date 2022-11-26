import sys
sys.path.append('/home/lishuang/evox')

import jax
import jax.numpy as jnp
import evox as ex
import gym
from PIL import Image

from Ant import AntPolicy as ProblemPolicy

model_path = '/home/lishuang/evox/logs/2022_11_15_10_23_38/best.npy'
gym_env = "Ant-v4"
seed = 42

def initialize_agent_policy():
    key = jax.random.PRNGKey(seed)
    agent_policy = ProblemPolicy()
    initial_params = agent_policy.init(key, jnp.zeros(agent_policy.observation_shape)) 

    adapter = ex.utils.TreeAndVector(initial_params)
    policy_gene = jnp.load(model_path)
    zeros_gene = jnp.zeros(policy_gene.shape)
    print(zeros_gene)
    # policy_params = adapter.to_tree(policy_gene)
    policy_params = adapter.to_tree(zeros_gene)
    
    return lambda observation: agent_policy.apply(policy_params, observation)

def evaluate(env, policy, collect_rgbarray=False):
    observation, info = env.reset(seed=0)
    total_reward, eposide_length = 0, 0
    rgb_arrays = []
    while True:
        if collect_rgbarray:
            rgb_arrays.append(env.render())

        action = policy(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        print(terminated, truncated)
        total_reward += reward
        eposide_length += 1
        if terminated or truncated:
            break

    if collect_rgbarray:
        return total_reward, rgb_arrays
    else:
        return total_reward

def rgb_arrays2gif(rgb_arrays, saveName, duration=None, loop=0, fps=None):
    imgs = [Image.fromarray(e) for e in rgb_arrays]
    if fps:
        duration = 1 / fps
    duration *= 1000
    imgs[0].save(saveName, save_all=True, append_images=imgs, duration=duration, loop=loop)

def main():
    policy = initialize_agent_policy()
    env = gym.make("Ant-v4", render_mode="rgb_array")
    reward, rgb_arrays = evaluate(env, policy, collect_rgbarray=True)

    rgb_arrays2gif(rgb_arrays, "./videos/0.gif", duration=0.3)

    print(reward)
    env.close()

if __name__ == "__main__":
    main()

