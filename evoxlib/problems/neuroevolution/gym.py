from functools import reduce
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import ray
import evoxlib as exl
from evoxlib import Problem, Module, State

import gym


@ray.remote(num_cpus=1)
class Worker:
    def __init__(self, env_name, num_env):
        self.num_env = num_env
        self.envs = [gym.make(env_name) for _ in range(num_env)]

    def step(self, actions):
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            # take the action if not terminated
            if not self.terminated[i]:
                self.observations[i], reward, self.terminated[i], _truncated = env.step(
                    action
                )
                self.total_rewards[i] += reward
                self.episode_length[i] += 1
        return self.observations, self.terminated

    def get_rewards(self):
        return self.total_rewards

    def get_episode_length(self):
        return self.episode_length

    def reset(self):
        self.total_rewards = [0 for _ in range(self.num_env)]
        self.terminated = [False for _ in range(self.num_env)]
        self.observations = [env.reset() for env in self.envs]
        self.episode_length = [0 for _ in range(self.num_env)]
        return self.observations


@ray.remote
class Controller:
    def __init__(
        self,
        policy,
        num_workers,
        env_per_worker,
        env_name,
        worker_options={},
    ):
        self.num_workers = num_workers
        self.env_per_worker = env_per_worker
        self.workers = [
            Worker.options(**worker_options).remote(env_name, env_per_worker)
            for _ in range(num_workers)
        ]
        self.policy = policy

    def evaluate(self, pop, cap_episode_length):
        observations = ray.get([worker.reset.remote() for worker in self.workers])
        terminated = False
        episode_length = 0
        while not terminated and (cap_episode_length is None or episode_length < cap_episode_length):
            episode_length += 1
            observations = jnp.array(observations)
            # get action from policy
            actions = jax.vmap(self.policy)(pop, observations.reshape((self.num_workers*self.env_per_worker, -1)))
            # reshape in order to distribute to different workers
            actions = actions.reshape((self.num_workers, self.env_per_worker, -1))
            # convert to numpy array and squeeze if needed
            actions = np.array(actions)
            if actions.shape[2] == 1:
                actions = actions.squeeze(axis=2)

            futures = [
                worker.step.remote(action)
                for worker, action in zip(self.workers, actions)
            ]
            observations, terminated = zip(*ray.get(futures))
            # print("sum of terminated", reduce(lambda carry, elem: carry + sum(elem), terminated, 0))
            terminated = reduce(lambda carry, elem: carry and all(elem), terminated, True)

        rewards = [worker.get_rewards.remote() for worker in self.workers]
        episode_length = [worker.get_episode_length.remote() for worker in self.workers]
        rewards = ray.get(rewards)
        episode_length = ray.get(episode_length)

        return jnp.array(rewards).reshape(-1), jnp.array(episode_length).reshape(-1)


@exl.jit_class
class CapEpisode(Module):
    def __init__(self, init_cap=100):
        self.init_cap = init_cap

    def setup(self, key):
        return State(
            cap=self.init_cap
        )

    def update(self, state, episode_length):
        return state.update(
            cap=jnp.rint(jnp.mean(episode_length) * 2)
        )

    def get(self, state):
        return state, state.cap


class Gym(Problem):
    def __init__(
        self,
        policy: Callable,
        num_workers: int,
        env_per_worker: int,
        env_name: str = "CartPole-v1",
        controller_options: dict = {},
        worker_options: dict = {},
        cap_episode: Module = CapEpisode(),
        fitness_is_neg_reward: bool = True
    ):
        """Construct a gym problem

        Parameters
        ----------
        policy
            A function that accept two arguments
            the first one is the parameter and the second is the input.
        num_workers
            Number of worker actors.
        env_per_worker
            Number of gym environment per worker.
        env_name
            The name of the gym environment.
        controller_options
            The runtime options for controller actor.
            This actor is used to control workers and run the policy at each step.
            For example, to enable GPU acceleration on the policy network,
            set this field to::

                {"num_gpus": 1}

        worker_options
            The runtime options for worker actors.
        fitness_is_neg_reward
            If True, the fitness is the negative of the total reward,
            otherwise return the original reward.
        """
        self.controller = Controller.options(**controller_options).remote(
            policy, num_workers, env_per_worker, env_name, worker_options
        )
        self.fitness_is_neg_reward = fitness_is_neg_reward
        self.env_name = env_name
        self.policy = policy
        self.cap_episode = cap_episode

    def evaluate(self, state, pop):
        cap_episode_length = None
        if self.cap_episode:
            state, cap_episode_length = self.cap_episode.get(state)

        rewards, episode_length = ray.get(self.controller.evaluate.remote(pop, cap_episode_length))

        if self.cap_episode:
            state = self.cap_episode.update(state, episode_length)

        if self.fitness_is_neg_reward:
            return state, -rewards
        else:
            return state, rewards

    def _render(self, individual):
        env = gym.make(self.env_name)
        observation = env.reset()
        env.render()
        terminated = False
        while not terminated:
            observation = jnp.array(observation)
            action = self.policy(individual, observation)
            action = np.array(action)
            observation, reward, terminated, _truncated = env.step(action)
            env.render()
