from typing import Callable, Any
from brax import envs
from brax.io import html, image
import jax
from jax import jit, vmap
import jax.numpy as jnp
import jax.tree_util as jtu
from evox import Problem, State, jit_method


def vmap_rng_split(key: jax.Array, num: int = 2) -> jax.Array:
    # batched_key [B, 2] -> batched_keys [num, B, 2]
    return jax.vmap(jax.random.split, in_axes=(0, None), out_axes=1)(key, num)


class Brax(Problem):
    def __init__(
        self,
        policy: Callable,
        env_name: str,
        max_episode_length: int,
        num_episodes: int,
        stateful_policy: bool = False,
        initial_state: Any = None,
        reduce_fn: Callable[[jax.Array, int], jax.Array] = jnp.mean,
        backend: str = "generalized",
    ):
        """Contruct a brax-based problem

        Parameters
        ----------
        policy
            a callable: fn(weights, obs) -> action
        env_name
            The environment name.
        batch_size
            The number of brax environments to run in parallel.
            Usually this should match the population size at the algorithm side.
        max_episode_length
            The maximum number of timesteps of an episode.
        num_episodes
            Evaluating the number of episodes for each individual.
        stateful_policy
            Whether the policy is stateful (for example, RNN).
            Default to False.
            If False, the policy should be a pure function with signature (weights, obs) -> action.
            If True, the policy should be a stateful function with signature (state, weights, obs) -> (action, state).
        initial_state
            The initial state of the stateful policy.
            Default to None.
            Only used when stateful_policy is True.
        reduce_fn
            The function to reduce the rewards of multiple episodes.
            Default to jnp.mean.
        backend
            Brax's backend, one of "generalized", "positional", "spring".
            Default to "generalized".
        """
        self.batched_policy = jit(vmap(vmap(policy, in_axes=(None, 0))))
        self.policy = policy
        self.env_name = env_name
        self.backend = backend
        self.env = envs.wrappers.training.VmapWrapper(
            envs.get_environment(env_name=env_name, backend=backend)
        )
        self.stateful_policy = stateful_policy
        self.initial_state = initial_state
        self.max_episode_length = max_episode_length
        self.num_episodes = num_episodes
        self.reduce_fn = reduce_fn

        self.jit_reset = jit(vmap(self.env.reset))
        self.jit_env_step = jit(vmap(self.env.step))

    def setup(self, key):
        return State(key=key)

    @jit_method
    def evaluate(self, state, weights):
        pop_size = jtu.tree_leaves(weights)[0].shape[0]
        key, eval_key = jax.random.split(state.key)

        def _cond_func(carry):
            counter, _state, done, _total_reward = carry
            return (counter < self.max_episode_length) & (~done.all())

        def _body_func(carry):
            counter, rollout_state, done, total_reward = carry
            if self.stateful_policy:
                state, brax_state = rollout_state
                action, state = self.batched_policy(state, weights, brax_state.obs)
                rollout_state = (state, brax_state)
            else:
                (brax_state,) = rollout_state
                action = self.batched_policy(weights, brax_state.obs)
                rollout_state = (brax_state,)
            brax_state = self.jit_env_step(brax_state, action)
            done = brax_state.done * (1 - done)
            total_reward += (1 - done) * brax_state.reward
            return counter + 1, rollout_state, done, total_reward

        brax_state = self.jit_reset(
            vmap_rng_split(jax.random.split(eval_key, self.num_episodes), pop_size)
        )

        if self.stateful_policy:
            rollout_state = (self.initial_state, brax_state)
        else:
            rollout_state = (brax_state,)

        # [pop_size, num_episodes]
        _, _, _, total_reward = jax.lax.while_loop(
            _cond_func,
            _body_func,
            (
                0,
                rollout_state,
                jnp.zeros((pop_size, self.num_episodes)),
                jnp.zeros((pop_size, self.num_episodes)),
            ),
        )

        total_reward = self.reduce_fn(total_reward, axis=-1)

        return total_reward, state.replace(key=key)

    def visualize(
        self,
        key,
        weights,
        output_type: str = "HTML",
        respect_done=False,
        *args,
        **kwargs,
    ):
        assert output_type in [
            "HTML",
            "rgb_array",
        ], "output_type must be either HTML or rgb_array"

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

            if respect_done and brax_state.done:
                break

        if output_type == "HTML":
            return html.render(env.sys.replace(dt=env.dt), trajectory, *args, **kwargs)
        else:
            return [
                image.render_array(sys=self.env.sys, state=s, **kwargs)
                for s in trajectory
            ]
