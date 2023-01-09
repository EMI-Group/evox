from functools import reduce
from typing import Callable, Optional

import gym
import jax
import jax.numpy as jnp
import numpy as np
import ray
from jax import jit, vmap
from jax.tree_util import tree_map, tree_structure, tree_transpose

import evox as ex
from evox import Problem, State, Stateful


@ex.jit_class
class Normalizer(Stateful):
    def __init__(self):
        self.sum = 0
        self.sumOfSquares = 0
        self.count = 0

    def setup(self, key):
        return ex.State(sum=0, sumOfSquares=0, count=0)

    def normalize(self, state, x):
        newCount = state.count + 1
        newSum = state.sum + x
        newSumOfSquares = state.sumOfSquares + x**2
        state = state.update(count=newCount, sum=newSum, sumOfSquares=newSumOfSquares)
        state, mean = self.mean(state)
        state, std = self.std(state)
        return state, (x - mean) / std

    def mean(self, state):
        mean = state.sum / state.count
        return state.update(mean=mean), mean

    def std(self, state):
        return state, jnp.sqrt(
            jnp.maximum(state.sumOfSquares / state.count - state.mean**2, 1e-2)
        )

    def normalize_obvs(self, state, obvs):

        newCount = state.count + len(obvs)
        newSum = state.sum + jnp.sum(obvs, axis=0)
        newSumOFsquares = state.sumOfSquares + jnp.sum(obvs**2, axis=0)
        state = state.update(count=newCount, sum=newSum, sumOfSquares=newSumOFsquares)

        state, mean = self.mean(state)
        state, std = self.std(state)
        return state, (obvs - mean) / std


@ray.remote(num_cpus=1)
class Worker:
    def __init__(self, env_name, env_options, num_env, policy=None):
        self.num_env = num_env
        self.envs = [gym.make(env_name, **env_options) for _ in range(num_env)]
        self.policy = policy

        self.seed2key = jit(vmap(jax.random.PRNGKey))
        self.splitKey = jit(vmap(jax.random.split))

    def step(self, actions):
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            # take the action if not terminated
            if not (self.terminated[i] or self.truncated[i]):
                (
                    self.observations[i],
                    reward,
                    self.terminated[i],
                    self.truncated[i],
                    self.infos[i],
                ) = env.step(action)

                self.total_rewards[i] += reward
                self.episode_length[i] += 1

        return self.observations, self.terminated, self.truncated

    def get_rewards(self):
        return self.total_rewards

    def get_episode_length(self):
        return self.episode_length

    def reset(self, seeds):
        self.total_rewards = np.zeros((self.num_env,))
        self.episode_length = np.zeros((self.num_env,))
        self.terminated = np.zeros((self.num_env,), dtype=bool)
        self.truncated = np.zeros((self.num_env,), dtype=bool)
        self.observations, self.infos = zip(
            *[env.reset(seed=seed) for seed, env in zip(seeds, self.envs)]
        )
        self.observations, self.infos = list(self.observations), list(self.infos)
        return self.observations

    def rollout(self, seeds, subpop, cap_episode_length):
        assert self.policy is not None
        self.reset(seeds)
        i = 0
        while True:
            observations = jnp.asarray(self.observations)
            actions = np.asarray(
                self.policy(subpop, observations)
            )
            self.step(actions)

            if np.all(self.terminated | self.truncated):
                break

            i += 1
            if cap_episode_length and i >= cap_episode_length:
                break

        return self.total_rewards, self.episode_length


@ray.remote
class Controller:
    def __init__(
        self,
        policy,
        num_workers,
        env_per_worker,
        env_name,
        env_options,
        worker_options,
        batch_policy,
    ):
        self.num_workers = num_workers
        self.env_per_worker = env_per_worker
        self.workers = [
            Worker.options(**worker_options).remote(
                env_name,
                env_options,
                env_per_worker,
                None if batch_policy else jit(vmap(policy)),
            )
            for _ in range(num_workers)
        ]
        self.policy = policy
        self.batch_policy = batch_policy

    def _evaluate(self, seeds, pop, cap_episode_length):
        def slice_pop(pop):
            def reshape_weight(w):
                # first dim is batch
                weight_dim = w.shape[1:]
                return list(
                    w.reshape((self.num_workers, self.env_per_worker, *weight_dim))
                )

            if isinstance(pop, jnp.ndarray):
                # first dim is batch
                param_dim = pop.shape[1:]
                pop = pop.reshape((self.num_workers, self.env_per_worker, *param_dim))
            else:
                outer_treedef = tree_structure(pop)
                inner_treedef = tree_structure([0 for i in range(self.num_workers)])
                pop = tree_map(reshape_weight, pop)
                pop = tree_transpose(outer_treedef, inner_treedef, pop)

            return pop

        sliced_pop = jit(slice_pop)(pop)
        rollout_future = [
            worker.rollout.remote(worker_seeds, subpop, cap_episode_length)
            for worker_seeds, subpop, worker in zip(seeds, sliced_pop, self.workers)
        ]

        rewards, episode_length = zip(*ray.get(rollout_future))
        return np.array(rewards).reshape(-1), np.array(episode_length).reshape(-1)

    def _batched_evaluate(self, seeds, pop, cap_episode_length):
        observations = ray.get(
            [
                worker.reset.remote(worker_seeds)
                for worker_seeds, worker in zip(seeds, self.workers)
            ]
        )
        terminated = False
        episode_length = 0

        def batch_policy_evaluation(observations):
            # the first two dims are num_workers and env_per_worker
            observation_dim = observations.shape[2:]
            actions = jax.vmap(self.policy)(
                pop,
                observations.reshape(
                    (self.num_workers * self.env_per_worker, *observation_dim)
                ),
            )
            # reshape in order to distribute to different workers
            action_dim = actions.shape[1:]
            actions = actions.reshape(
                (self.num_workers, self.env_per_worker, *action_dim)
            )
            return actions

        i = 0
        while True:
            observations = jnp.asarray(observations)
            # get action from policy
            actions = jit(batch_policy_evaluation)(observations)
            # convert to numpy array
            actions = np.asarray(actions)

            futures = [
                worker.step.remote(action)
                for worker, action in zip(self.workers, actions)
            ]
            observations, terminated = zip(*ray.get(futures))

            all_terminated = reduce(
                lambda carry, elem: carry and all(elem), terminated, True
            )

            if all_terminated:
                break

            i += 1
            if cap_episode_length and i >= cap_episode_length:
                break

        rewards = [worker.get_rewards.remote() for worker in self.workers]
        episode_length = [worker.get_episode_length.remote() for worker in self.workers]
        rewards = ray.get(rewards)
        episode_length = ray.get(episode_length)
        return np.array(rewards).reshape(-1), np.array(episode_length).reshape(-1)

    def evaluate(self, seeds, pop, cap_episode_length):
        if self.batch_policy:
            return self._batched_evaluate(seeds, pop, cap_episode_length)
        else:
            return self._evaluate(seeds, pop, cap_episode_length)


@ex.jit_class
class CapEpisode(Stateful):
    def __init__(self, init_cap=100):
        self.init_cap = init_cap

    def setup(self, key):
        return State(cap=self.init_cap)

    def update(self, state, episode_length):
        return state.update(
            cap=jnp.rint(jnp.mean(episode_length) * 2).astype(jnp.int32)
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
        env_options: dict = {},
        controller_options: dict = {},
        worker_options: dict = {},
        init_cap: Optional[int] = None,
        batch_policy: bool = False,
        fitness_is_neg_reward: bool = True,
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
            policy,
            num_workers,
            env_per_worker,
            env_name,
            env_options,
            worker_options,
            batch_policy,
        )
        self.num_workers = num_workers
        self.env_per_worker = env_per_worker
        self.fitness_is_neg_reward = fitness_is_neg_reward
        self.env_name = env_name
        self.policy = policy
        if init_cap is not None:
            self.cap_episode = CapEpisode(init_cap=init_cap)
        else:
            self.cap_episode = None

    def setup(self, key):
        return State(key=key)

    def evaluate(self, state, pop):
        key, subkey = jax.random.split(state.key)
        # generate a list of seeds for gym
        seeds = jax.random.randint(
            subkey, (self.num_workers, self.env_per_worker), 0, jnp.iinfo(jnp.int32).max
        )
        # seeds = jnp.full((self.num_workers, self.env_per_worker), 42)

        seeds = seeds.tolist()

        cap_episode_length = None
        if self.cap_episode:
            state, cap_episode_length = self.cap_episode.get(state)
            cap_episode_length = cap_episode_length.item()

        rewards, episode_length = ray.get(
            self.controller.evaluate.remote(seeds, pop, cap_episode_length)
        )

        # convert np.array -> jnp.array here
        # to avoid coping between cpu and gpu
        rewards = jnp.asarray(rewards)
        episode_length = jnp.asarray(episode_length)

        if self.cap_episode:
            state = self.cap_episode.update(state, episode_length)

        if self.fitness_is_neg_reward:
            fitness = -rewards
        else:
            fitness = rewards

        return state.update(key=key), fitness

    def _render(self, state, individual, ale_render_mode=None):
        key, subkey = jax.random.split(state.key)
        seed = jax.random.randint(subkey, (1,), 0, jnp.iinfo(jnp.int32).max).item()
        if ale_render_mode is None:
            env = gym.make(self.env_name)
        else:
            env = gym.make(self.env_name, render_mode=ale_render_mode)

        observation = env.reset(seed=seed)
        if ale_render_mode is None:
            env.render()
        terminated = False
        while not terminated:
            observation = jnp.array(observation)
            action = self.policy(individual, observation)
            action = np.array(action)
            observation, reward, terminated, _truncated = env.step(action)
            if ale_render_mode is None:
                env.render()

        return state.update(key=key)
