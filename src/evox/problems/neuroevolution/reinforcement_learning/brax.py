from typing import Callable, Any
from brax import envs
from brax.io import html, image
import jax
from jax import jit, vmap
import jax.numpy as jnp
from evox import Problem, State, jit_method


class Brax(Problem):
    def __init__(
        self,
        policy: Callable,
        env_name: str,
        batch_size: int,
        cap_episode: int,
        backend: str = "generalized",
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
        backend
            Brax's backend, one of "generalized", "positional", "spring".
            Default to "generalized".
        fitness_is_neg_reward
            Whether to return the fitness value as the negative of reward or not.
            Default to True.
        """
        self.batched_policy = jit(vmap(policy))
        self.policy = policy
        self.env_name = env_name
        self.backend = backend
        self.env = envs.wrappers.training.VmapWrapper(
            envs.get_environment(env_name=env_name, backend=backend)
        )
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

    def visualize(
        self, state, key, weights, output_type: str = "HTML", *args, **kwargs
    ):
        env = envs.get_environment(env_name=self.env_name, backend=self.backend)
        brax_state = jax.jit(env.reset)(key)
        jit_env_step = jit(env.step)
        trajectory = [brax_state.pipeline_state]
        episode_length = 1
        for _ in range(self.cap_episode):
            action = self.policy(weights, brax_state.obs)
            brax_state = jit_env_step(brax_state, action)
            trajectory.append(brax_state.pipeline_state)
            episode_length += 1 - brax_state.done

            if brax_state.done:
                break

        if output_type == "HTML":
            return (
                html.render(env.sys.replace(dt=env.dt), trajectory, *args, **kwargs),
                state,
            )
        else:
            return (
                image.render(env.sys.replace(dt=env.dt), trajectory, *args, **kwargs),
                state,
            )
