from typing import Callable, Any
from functools import partial
import brax
from brax import envs
from brax.io import html, image
import jax
from jax import jit, vmap
import jax.numpy as jnp
from jax.tree_util import tree_map
from evox import Problem, State, Stateful, jit_method


class Brax(Problem):
    def __init__(
        self,
        policy: Callable,
        env_name: str,
        batch_size: int,
        cap_episode: int,
        fitness_is_neg_reward: bool = True,
    ):
        """Contruct a brax-based problem

        Parameters
        ----------
        policy
            A function that accept two arguments
            the first one is the parameter and the second is the input.
        env_name
            The environment name.
        batch_size
            The number of brax environments to run in parallel.
            Usually this should match the population size at the algorithm side.
        cap_episode
            The maximum number episodes to run.
        fitness_is_neg_reward
            Whether to return the fitness value as the negative of reward or not.
            Default to True.
        """
        self.batched_policy = jit(vmap(policy))
        self.env = envs.wrappers.VmapWrapper(envs.create(env_name=env_name))
        self.batch_size = batch_size
        self.cap_episode = cap_episode
        self.fitness_is_neg_reward = fitness_is_neg_reward
        self.jit_reset = jit(self.env.reset)
        self.jit_env_step = jit(self.env.step)

    def setup(self, key):
        return State(init_state=self.jit_reset(jnp.tile(key, (self.batch_size, 1))))

    @jit_method
    def evaluate(self, state, weights):
        brax_state = state.init_state

        def cond_func(val):
            counter, state, _total_reward = val
            return (counter < self.cap_episode) & (~state.done.all())

        def body_func(val):
            counter, brax_state, total_reward = val
            action = self.batched_policy(weights, brax_state.obs)
            brax_state = self.jit_env_step(brax_state, action)
            total_reward += (1 - brax_state.done) * brax_state.reward
            return counter + 1, brax_state, total_reward

        init_val = (0, brax_state, jnp.zeros((self.batch_size,)))

        _counter, _brax_state, total_reward = jax.lax.while_loop(
            cond_func, body_func, init_val
        )

        if self.fitness_is_neg_reward:
            total_reward = -total_reward

        return total_reward, state

    def visualize(self, state, weights, output_type: str = "HTML", *args, **kwargs):
        brax_state = state.init_state
        trajectories = [brax_state.qp]
        episode_length = 1
        for _ in range(self.cap_episode):
            action = self.batched_policy(weights, brax_state.obs)
            brax_state = self.jit_env_step(brax_state, action)
            trajectories.append(brax_state.qp)
            episode_length += 1 - brax_state.done

            if brax_state.done.all():
                break

        # trajectories is now [batch_qp_0, ..., batch_qp_n-1]
        @jit
        def pytree_first_dim_to_list(obj):
            return [tree_map(lambda x: x[i], obj) for i in range(self.batch_size)]
        # slice through the batch dim
        # trajectories is now [[qp_0_0, qp_0_1, ...], ...]
        trajectories = [
            pytree_first_dim_to_list(qp) for qp in trajectories
        ]
        # transpose, make the batch_dim the first dim
        trajectories = list(zip(*trajectories))
        # strip out the states that are 'done'
        episode_length = episode_length.astype(int)
        trajectories = [
            trajectory[: length + 1]
            for trajectory, length in zip(trajectories, episode_length)
        ]
        if output_type == "HTML":
            return [
                html.render(self.env.sys, t, *args, **kwargs)
                for t in trajectories
            ], state
        else:
            return [
                image.render(self.env.sys, t, *args, **kwargs)
                for t in trajectories
            ], state
