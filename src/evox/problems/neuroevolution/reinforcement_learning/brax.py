from typing import Callable, Any, Optional
from brax import envs
from brax.io import html, image
import jax
from jax import jit, vmap
import jax.numpy as jnp
import jax.tree_util as jtu
from evox import Problem, State, jit_method


class Brax(Problem):
    def __init__(
        self,
        policy: Callable,
        env_name: str,
        max_episode_length: int,
        num_episodes: int,
        rotate_key: bool = True,
        stateful_policy: bool = False,
        initial_state: Any = None,
        reduce_fn: Callable[[jax.Array, int], jax.Array] = jnp.mean,
        backend: str = "generalized",
    ):
        """Contruct a brax-based problem.
        Firstly, you need to define a jit-able policy function. The policy function should have the following signature:
        If you policy is not stateful:
        :code:`fn(weights, obs) -> action`,
        and if you policy is stateful:
        :code:`fn(state, weights, obs) -> action, state`.
        Then you need to set the `environment name <https://github.com/google/brax/tree/main/brax/envs>`_,
        the maximum episode length, the number of episodes to evaluate for each individual.
        For each individual,
        it will run the policy with the environment for num_episodes times with different seed,
        and use the reduce_fn to reduce the rewards (default to average).
        Different individuals will share the same set of random keys in each iteration.

        Parameters
        ----------
        policy
            A callable if stateful: :code:`fn(state, weight, obs) -> action, state` otherwise :code:`fn(weights, obs) -> action`
        env_name
            The environment name.
        max_episode_length
            The maximum number of timesteps of each episode.
        num_episodes
            The number of episodes to evaluate for each individual.
        rotate_key
            Indicates whether to rotate the random key for each iteration (default is True).

            If True, the random key will rotate after each iteration,
            resulting in non-deterministic and potentially noisy fitness evaluations.
            This means that identical policy weights may yield different fitness values across iterations.

            If False, the random key remains the same for all iterations,
            ensuring consistent fitness evaluations.

        stateful_policy
            Whether the policy is stateful (for example, RNN).
            Default to False.
            If False, the policy should be a pure function with signature :code:`fn(weights, obs) -> action`.
            If True, the policy should be a stateful function with signature :code:`fn(state, weight, obs) -> action, state`.
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

        Notes
        -----
        When rotating keys, fitness evaluation is non-deterministic and may introduce noise.

        Examples
        --------
        >>> from evox import problems
        >>> problem = problems.neuroevolution.Brax(
        ...    env_name="swimmer",
        ...    policy=jit(model.apply),
        ...    max_episode_length=1000,
        ...    num_episodes=3,
        ...    rotate_key=False,
        ...)
        """
        if stateful_policy:
            self.batched_policy = jit(vmap(vmap(policy, in_axes=(0, None, 0))))
        else:
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
        self.rotate_key = rotate_key
        self.reduce_fn = reduce_fn

        self.jit_reset = jit(vmap(self.env.reset))
        self.jit_env_step = jit(vmap(self.env.step))

    def setup(self, key):
        return State(key=key)

    @jit_method
    def evaluate(self, state, weights):
        pop_size = jtu.tree_leaves(weights)[0].shape[0]
        if self.rotate_key:
            key, eval_key = jax.random.split(state.key)
        else:
            key, eval_key = state.key, state.key

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

        # For each episode, we need a different random key.
        keys = jax.random.split(eval_key, self.num_episodes)
        # For each individual in the population, we need the same set of keys.
        keys = jnp.broadcast_to(keys, (pop_size, *keys.shape))
        brax_state = self.jit_reset(keys)

        if self.stateful_policy:
            initial_state = jax.tree.map(
                lambda x: jnp.broadcast_to(x, (pop_size, self.num_episodes, *x.shape)),
                self.initial_state,
            )
            rollout_state = (initial_state, brax_state)
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
        respect_done: bool = False,
        num_episodes: Optional[int] = None,
        *args,
        **kwargs,
    ):
        """Visualize the brax environment with the given policy and weights.

        Parameters
        ----------
        key
            The random key.
        weights
            The weights of the policy.
        output_type
            The output type, either "HTML" or "rgb_array".
        respect_done
            Whether to respect the done signal.
        num_episodes
            The number of episodes to visualize, used to override the num_episodes in the constructor.
            If None, use the num_episodes in the constructor.
        """
        assert output_type in [
            "HTML",
            "rgb_array",
        ], "output_type must be either HTML or rgb_array"

        env = envs.get_environment(env_name=self.env_name, backend=self.backend)
        brax_state = jax.jit(env.reset)(key)
        jit_env_step = jit(env.step)
        trajectory = [brax_state.pipeline_state]
        episode_length = 1

        if self.stateful_policy:
            rollout_state = (self.initial_state, brax_state)
        else:
            rollout_state = (brax_state,)

        num_episodes = num_episodes or self.num_episodes
        for _ in range(num_episodes):
            if self.stateful_policy:
                state, brax_state = rollout_state
                action, state = self.policy(state, weights, brax_state.obs)
                rollout_state = (state, brax_state)
            else:
                (brax_state,) = rollout_state
                action = self.policy(weights, brax_state.obs)
                rollout_state = (brax_state,)

            trajectory.append(brax_state.pipeline_state)
            brax_state = jit_env_step(brax_state, action)
            trajectory.append(brax_state.pipeline_state)
            episode_length += 1 - brax_state.done

            if respect_done and brax_state.done:
                break

        if output_type == "HTML":
            return html.render(env.sys, trajectory, *args, **kwargs)
        else:
            return image.render_array(env.sys, trajectory, **kwargs)
