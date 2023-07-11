from typing import Callable

import gym
import jax
import jax.numpy as jnp
import numpy as np

from evox import Problem, State


class Gym(Problem):
    def __init__(
            self,
            pop_size: int,
            policy: Callable,
            env_name: str = "CartPole-v1",
            env_options: dict = None,
            batch_policy: bool = True,
    ):
        self.pop_size = pop_size
        self.env_name = env_name
        self.policy = policy
        self.env_options = env_options or {}
        self.batch_policy = batch_policy
        assert batch_policy, "Only batch policy is supported for now"

        self.envs = [gym.make(env_name, **self.env_options) for _ in range(self.pop_size)]

        super().__init__()

    def setup(self, key):
        return State(key=key)

    def evaluate(self, state, pop):
        key = state.key
        seeds = jax.random.randint(
            key, (self.pop_size,), 0, jnp.iinfo(jnp.int32).max
        )

        seeds = seeds.tolist()  # seed must be a python int, not numpy array
        print(seeds)
        fitnesses = self.__rollout(seeds, pop)
        print("fitnesses info: ")
        print(f"max: {np.max(fitnesses)}, min: {np.min(fitnesses)}, mean: {np.mean(fitnesses)}, std: {np.std(fitnesses)}")

        return -fitnesses, State(key=key)

    def __rollout(self, seeds, pop):
        observations = [env.reset(seed=seed) for env, seed in zip(self.envs, seeds)]
        terminates, truncates = np.zeros((2, self.pop_size), dtype=bool)
        fitnesses, rewards = np.zeros((2, self.pop_size))

        while not np.all(terminates | truncates):
            observations = np.asarray(observations)
            actions = self.policy(pop, observations)
            actions = jax.device_get(actions)

            for i, (action, terminate, truncate, env) in enumerate(zip(actions, terminates, truncates, self.envs)):
                if terminate | truncate:
                    observation = np.zeros(env.observation_space.shape)
                    reward = 0
                else:
                    observation, reward, terminate, truncate, info = env.step(action)

                observations[i] = observation
                rewards[i] = reward
                terminates[i] = terminate
                truncates[i] = truncate

            fitnesses += rewards

        return fitnesses
