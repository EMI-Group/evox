from typing import Callable, Optional, List

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import ray
from jax import jit, vmap
from jax.tree_util import tree_map, tree_structure, tree_transpose, tree_leaves

from evox import Problem, State, Stateful, jit_class, jit_method


@jit
def tree_batch_size(tree):
    """Get the batch size of a tree"""
    return tree_leaves(tree)[0].shape[0]


@jit_class
class Normalizer(Stateful):
    def __init__(self):
        self.sum = 0
        self.sumOfSquares = 0
        self.count = 0

    def setup(self, key):
        return State(sum=0, sumOfSquares=0, count=0)

    def normalize(self, state, x):
        newCount = state.count + 1
        newSum = state.sum + x
        newSumOfSquares = state.sumOfSquares + x**2
        state = state.update(count=newCount, sum=newSum, sumOfSquares=newSumOfSquares)
        mean, state = self.mean(state)
        std, state = self.std(state)
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

        mean, state = self.mean(state)
        std, state = self.std(state)
        return state, (obvs - mean) / std


@ray.remote(num_cpus=1)
class Worker:
    def __init__(self, env_creator, policy=None, mo_keys=None):
        self.envs = []
        self.env_creator = env_creator
        self.policy = policy
        self.mo_keys = mo_keys

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

                mo_values = []
                for key in self.mo_keys:
                    if key not in self.infos[i]:
                        raise KeyError(
                            (
                                f"mo_keys has a key {key}, "
                                "which doesn't exist in `info`, "
                                f"available fields are {list(self.infos[i].keys())}."
                            )
                        )
                    mo_values.append(self.infos[i][key])
                mo_values = np.array(mo_values)
                self.acc_mo_values += mo_values

        return self.observations, self.terminated, self.truncated

    def get_rewards(self):
        return self.total_rewards, self.acc_mo_values

    def get_episode_length(self):
        return self.episode_length

    def reset(self, seed, num_env):
        # create new envs if needed
        while len(self.envs) < num_env:
            self.envs.append(self.env_creator())

        self.total_rewards = np.zeros((num_env,))
        self.acc_mo_values = np.zeros((len(self.mo_keys),))  # accumulated mo_value
        self.episode_length = np.zeros((num_env,))
        self.terminated = np.zeros((num_env,), dtype=bool)
        self.truncated = np.zeros((num_env,), dtype=bool)
        self.observations, self.infos = zip(
            *[env.reset(seed=seed) for env in self.envs[:num_env]]
        )
        self.observations, self.infos = list(self.observations), list(self.infos)
        return self.observations

    def rollout(self, seed, subpop, cap_episode_length):
        subpop = jax.device_put(subpop)
        # num_env is the first dim of subpop
        num_env = tree_batch_size(subpop)
        assert self.policy is not None
        self.reset(seed, num_env)
        i = 0
        while True:
            observations = jnp.asarray(self.observations)
            actions = np.asarray(self.policy(subpop, observations))
            self.step(actions)

            if np.all(self.terminated | self.truncated):
                break

            i += 1
            if cap_episode_length and i >= cap_episode_length:
                break

        return self.total_rewards, self.acc_mo_values, self.episode_length


@ray.remote
class Controller:
    def __init__(
        self,
        policy,
        num_workers,
        env_creator,
        worker_options,
        batch_policy,
        mo_keys,
    ):
        self.num_workers = num_workers
        self.workers = [
            Worker.options(**worker_options).remote(
                env_creator,
                None if batch_policy else jit(vmap(policy)),
                mo_keys,
            )
            for _ in range(num_workers)
        ]
        self.policy = policy
        self.batch_policy = batch_policy
        self.num_obj = len(mo_keys)

    @jit_method
    def slice_pop(self, pop):
        def reshape_weight(w):
            # first dim is batch
            weight_dim = w.shape[1:]
            return jnp.array_split(w, self.num_workers, axis=0)

        if isinstance(pop, jax.Array):
            # first dim is batch
            param_dim = pop.shape[1:]
            pop = jnp.array_split(pop, self.num_workers, axis=0)
        else:
            outer_treedef = tree_structure(pop)
            inner_treedef = tree_structure([0 for _i in range(self.num_workers)])
            pop = tree_map(reshape_weight, pop)
            pop = tree_transpose(outer_treedef, inner_treedef, pop)

        return pop

    def _evaluate(self, seed, pop, cap_episode_length):
        sliced_pop = self.slice_pop(pop)
        rollout_future = [
            worker.rollout.remote(seed, subpop, cap_episode_length)
            for subpop, worker in zip(sliced_pop, self.workers)
        ]

        rewards, acc_mo_values, episode_length = zip(*ray.get(rollout_future))
        rewards = np.concatenate(rewards, axis=0)
        acc_mo_values = np.concatenate(acc_mo_values, axis=0)
        episode_length = np.concatenate(episode_length, axis=0)
        acc_mo_values = np.array(acc_mo_values)
        return rewards, acc_mo_values, episode_length

    @jit_method
    def batch_policy_evaluation(self, observations, pop):
        actions = jax.vmap(self.policy)(
            pop,
            observations,
        )
        # reshape in order to distribute to different workers
        action_dim = actions.shape[1:]
        actions = jnp.array_split(actions, self.num_workers, axis=0)
        return actions

    def _batched_evaluate(self, seed, pop, cap_episode_length):
        pop_size = tree_batch_size(pop)
        env_per_worker = pop_size // self.num_workers
        reminder = pop_size % self.num_workers
        num_envs = [
            env_per_worker + 1 if i < reminder else env_per_worker
            for i in range(self.num_workers)
        ]
        observations = ray.get(
            [
                worker.reset.remote(seed, num_env)
                for worker, num_env in zip(self.workers, num_envs)
            ]
        )
        terminated = False
        episode_length = 0

        i = 0
        while True:
            # flatten observations
            observations = [obs for worker_obs in observations for obs in worker_obs]
            observations = np.stack(observations, axis=0)
            observations = jnp.asarray(observations)
            # get action from policy
            actions = self.batch_policy_evaluation(observations, pop)

            futures = [
                worker.step.remote(np.asarray(action))
                for worker, action in zip(self.workers, actions)
            ]
            observations, terminated, truncated = zip(*ray.get(futures))
            terminated = np.concatenate(terminated, axis=0)
            truncated = np.concatenate(truncated, axis=0)
            if np.all(terminated | truncated):
                break

            i += 1
            if cap_episode_length and i >= cap_episode_length:
                break

        rewards, acc_mo_values = zip(
            *ray.get([worker.get_rewards.remote() for worker in self.workers])
        )
        rewards = np.concatenate(rewards, axis=0)
        acc_mo_values = np.concatenate(acc_mo_values, axis=0)
        episode_length = [worker.get_episode_length.remote() for worker in self.workers]
        episode_length = ray.get(episode_length)
        episode_length = np.concatenate(episode_length, axis=0)
        return rewards, acc_mo_values, episode_length

    def evaluate(self, seed, pop, cap_episode_length):
        if self.batch_policy:
            return self._batched_evaluate(seed, pop, cap_episode_length)
        else:
            return self._evaluate(seed, pop, cap_episode_length)


@jit_class
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
        return state.cap, state


class Gym(Problem):
    def __init__(
        self,
        policy: Callable,
        num_workers: int,
        env_name: Optional[str] = None,
        env_options: dict = {},
        env_creator: Optional[Callable] = None,
        mo_keys: List = [],
        controller_options: dict = {},
        worker_options: dict = {},
        init_cap: Optional[int] = None,
        batch_policy: bool = False,
    ):
        """Construct a gym problem

        Parameters
        ----------
        policy
            A function that accept two arguments
            the first one is the parameter and the second is the input.
        num_workers
            Number of worker actors.
        env_name
            The name of the gym environment.
        env_options
            The options of the gym environment.
        env_creator
            A function with zero argument that returns an environment when called.
        mo_keys
            Optional, a list of strings.
            If set, the environment is treated as a multi-objective problem,
            and different objective values are obtained through the `info` term returned by Gym.
            The `mo_keys` parameter provides the keys for accessing the objective values in the info dictionary.
            The objective values will be returned in the same order as specified in `mo_keys`.
        controller_options
            The runtime options for controller actor.
            This actor is used to control workers and run the policy at each step.
            For example, to enable GPU acceleration on the policy network,
            set this field to::

                {"num_gpus": 1}
        worker_options
            The runtime options for worker actors.
        """
        if env_name:
            env_creator = lambda: gym.make(env_name, **env_options)
        if not env_creator:
            raise ValueError("Either 'env_name' or 'env_creator' must be set.'")

        self.mo_keys = mo_keys
        self.controller = Controller.options(**controller_options).remote(
            policy,
            num_workers,
            env_creator,
            worker_options,
            batch_policy,
            mo_keys,
        )
        self.num_workers = num_workers
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
        seed = jax.random.randint(subkey, (1,), 0, jnp.iinfo(jnp.int32).max).item()

        cap_episode_length = None
        if self.cap_episode:
            cap_episode_length, state = self.cap_episode.get(state)
            cap_episode_length = cap_episode_length.item()

        rewards, acc_mo_values, episode_length = ray.get(
            self.controller.evaluate.remote(seed, pop, cap_episode_length)
        )

        # convert np.array -> jnp.array here
        # to avoid coping between cpu and gpu
        rewards = jnp.asarray(rewards)
        episode_length = jnp.asarray(episode_length)

        if self.cap_episode:
            state = self.cap_episode.update(state, episode_length)

        fitness = rewards

        if self.mo_keys:
            return acc_mo_values, state.update(key=key)
        else:
            return fitness, state.update(key=key)

    def visualize(self, state, key, weights, ale_render_mode="rgb_array"):
        """Visualize your policy, passin a single set of weights,
        and it will be put in the environment for interaction.

        Parameters
        ----------
        state
            The state.
        key
            This key will be used to seed the test environment.
        weights
            A single set of weights for your policy.
        ale_render_mode
            'rgb_array' or 'human'.

            In 'rgb_array' mode, this function will return a list of frames,
            each frame is a numpy array.

            In 'human' mode,
            the frame should be displayed directly onto your screen.
            However, if your using remote python environment, for example
            vscode ssh or jupyter notebook,
            this method may fail to find a valid display.
            Default to 'rgb_array'.
        """
        seed = jax.random.randint(key, (1,), 0, jnp.iinfo(jnp.int32).max).item()
        env = gym.make(self.env_name, render_mode=ale_render_mode)

        observation, info = env.reset(seed=seed)
        frames = []
        if ale_render_mode == "rgb_array":
            frames.append(observation)

        terminated = False
        while not terminated:
            observation = jnp.array(observation)
            action = self.policy(weights, observation)
            action = np.array(action)
            observation, _reward, terminated, _truncated, info = env.step(action)

            if ale_render_mode == "rgb_array":
                frames.append(observation)

        return frames, state
