import gym
env = gym.make("HalfCheetah-v4")
observation, info = env.reset(seed=42)
print(env)
print(env.observation_space)
print(env.action_space)
total_reward = 0

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    reward_run, reward_ctrl = info["reward_run"], info["reward_ctrl"]
    # survive, forward_distance, ctrl_cost = info["reward_survive"], info["reward_forward"], info["reward_ctrl"]
    cal_total_reward = reward_run + reward_ctrl
    assert reward == cal_total_reward
    print(reward, cal_total_reward)
    total_reward += reward
    if terminated or truncated:
        break

print(total_reward)
env.close()