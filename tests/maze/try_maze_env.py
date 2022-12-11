# import jax
# key = jax.random.PRNGKey(41)
# for i in range(100):
#     key, _ = jax.random.split(key)
#     if jax.random.uniform(key) < 0.1:
#         print(1)
#     else:
#         print(2)


import time
import gym

action_list = [2, 2, 0, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
action_it = iter(action_list)

env = gym.make('Maze_navigation-v0', maze_idx=0, time_limit=100, easy_mode=True)
observation, _ = env.reset(seed=42)
total_reward = 0

while True:
    action = next(action_it)
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    print(observation, reward)
    if terminated or truncated:
        observation, info = env.reset()
        break
env.close()
print(total_reward)
