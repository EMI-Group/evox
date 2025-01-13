__all__ = ["BraxProblem"]

import weakref
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import torch
import torch.nn as nn
import torch.utils.dlpack
from brax import envs

from ...core import Problem, jit_class
from .utils import get_vmap_model_state_forward


# to_dlpack is not necessary for torch.Tensor and jax.Array
# because they have a __dlpack__ method, which is called by their respective from_dlpack methods.
def to_jax_array(x: torch.Tensor) -> jax.Array:
    return jax.dlpack.from_dlpack(x.detach())


def from_jax_array(x: jax.Array) -> torch.Tensor:
    return torch.utils.dlpack.from_dlpack(x)


__brax_data__: Dict[
    int,
    Tuple[Callable[[jax.Array], envs.State], Callable[[envs.State, jax.Array], envs.State]],
] = {}  # Cannot be a weakref.WeakValueDictionary because the values are only stored here


@jit_class
class BraxProblem(Problem):
    """The Brax problem wrapper."""

    def __init__(
        self,
        policy: nn.Module,
        env_name: str,
        max_episode_length: int,
        num_episodes: int,
        pop_size: int | None = None,
        rotate_key: bool = True,
        reduce_fn: Callable[[torch.Tensor, int], torch.Tensor] = torch.mean,
        backend: str | None = None,
        device: torch.device | None = None,
    ):
        """Construct a Brax-based problem.
        Firstly, you need to define a policy model.
        Then you need to set the `environment name <https://github.com/google/brax/tree/main/brax/envs>`,
        the maximum episode length, the number of episodes to evaluate for each individual.
        For each individual,
        it will run the policy with the environment for num_episodes times with different seed,
        and use the reduce_fn to reduce the rewards (default to average).
        Different individuals will share the same set of random keys in each iteration.

        :param policy: The policy model whose forward function is :code:`forward(batched_obs) -> action`.
        :param env_name: The environment name.
        :param max_episode_length: The maximum number of time steps of each episode.
        :param num_episodes: The number of episodes to evaluate for each individual.
        :param pop_size: The size of the population to be evaluated. If None, we expect the input to have a population size of 1.
        :param rotate_key: Indicates whether to rotate the random key for each iteration (default is True). <br/> If True, the random key will rotate after each iteration, resulting in non-deterministic and potentially noisy fitness evaluations. This means that identical policy weights may yield different fitness values across iterations. <br/> If False, the random key remains the same for all iterations, ensuring consistent fitness evaluations.
        :param reduce_fn: The function to reduce the rewards of multiple episodes. Default to `torch.mean`.
        :param backend: Brax's backend. If None, the default backend of the environment will be used. Default to None.
        :param device: The device to run the computations on. Defaults to the current default device.

        ## Notice
        The initial key is obtained from `torch.random.get_rng_state()`.

        ## Warning
        This problem does NOT support HPO wrapper (`problems.hpo_wrapper.HPOProblemWrapper`), i.e., the workflow containing this problem CANNOT be vmapped.

        ## Examples
        >>> from evox import problems
        >>> problem = problems.neuroevolution.Brax(
        ...    env_name="swimmer",
        ...    policy=model,
        ...    max_episode_length=1000,
        ...    num_episodes=3,
        ...    pop_size=100,
        ...    rotate_key=False,
        ...)
        """
        super().__init__()
        device = torch.get_default_device() if device is None else device
        pop_size = 1 if pop_size is None else pop_size
        # Create Brax environment
        env: envs.Env = (
            envs.get_environment(env_name=env_name)
            if backend is None
            else envs.get_environment(env_name=env_name, backend=backend)
        )
        env = envs.wrappers.training.VmapWrapper(env)
        # Compile Brax environment
        brax_reset = jax.jit(env.reset)
        brax_step = jax.jit(env.step)
        global __brax_data__
        __brax_data__[id(self)] = (brax_reset, brax_step)
        weakref.finalize(self, __brax_data__.pop, id(self), None)
        self._index_id_ = id(self)
        # JIT stateful model forward
        dummy_obs = torch.empty(pop_size, num_episodes, env.observation_size, device=device)
        _, jit_vmap_state_forward, _, _, param_to_state_key_map, model_buffers = get_vmap_model_state_forward(
            model=policy,
            pop_size=pop_size,
            dummy_inputs=dummy_obs,
            check_output=lambda x: (
                isinstance(x, torch.Tensor)
                and x.ndim == 3
                and x.shape[0] == pop_size
                and x.shape[1] == num_episodes
                and x.shape[2] == env.action_size
            ),
            device=device,
            vmap_in_dims=(0, 0),
        )
        self._jit_state_forward = jit_vmap_state_forward
        self._param_to_state_key_map = param_to_state_key_map
        model_buffers["key"] = torch.random.get_rng_state().view(dtype=torch.uint32)[:2].detach().clone().to(device=device)
        self._model_buffers = model_buffers
        # Set constants
        self.max_episode_length = max_episode_length
        self.num_episodes = num_episodes
        self.pop_size = pop_size
        self.rotate_key = rotate_key
        self.reduce_fn = reduce_fn

    def evaluate(self, pop_params: Dict[str, nn.Parameter]) -> torch.Tensor:
        """Evaluate the final rewards of a population (batch) of model parameters.

        :param pop_params: A dictionary of parameters where each key is a parameter name and each value is a tensor of shape (batch_size, *param_shape) representing the batched parameters of batched models.

        :return: A tensor of shape (batch_size,) containing the reward of each sample in the population.
        """
        # Unpack parameters and buffers
        state_params = {self._param_to_state_key_map[key]: value for key, value in pop_params.items()}
        model_state = dict(self._model_buffers)
        model_state.update(state_params)
        rand_key = model_state.pop("key")
        # Brax environment evaluation
        model_state, rewards = self._evaluate_brax(model_state, rand_key)
        rewards = self.reduce_fn(rewards, dim=-1)
        # Update buffers
        self._model_buffers = {key: model_state[key] for key in self._model_buffers}
        # Return
        return rewards

    def _model_forward(
        self, model_state: Dict[str, torch.Tensor], obs: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        return self._jit_state_forward(model_state, obs)

    @torch.jit.ignore
    def _evaluate_brax(
        self, model_state: Dict[str, torch.Tensor], rand_key: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        # Get from torch
        pop_size = list(model_state.values())[0].shape[0]
        assert pop_size == self.pop_size
        global __brax_data__
        brax_reset, brax_step = __brax_data__[self._index_id_]
        # For each episode, we need a different random key.
        # For each individual in the population, we need the same set of keys.
        key = to_jax_array(rand_key)
        if self.rotate_key:
            key, eval_key = jax.random.split(key)
        else:
            key, eval_key = key, key
        keys = jax.random.split(eval_key, self.num_episodes)
        keys = jnp.broadcast_to(keys, (pop_size, *keys.shape)).reshape(pop_size * self.num_episodes, -1)
        # Loop until environment stops
        done = jnp.zeros((pop_size * self.num_episodes,), dtype=bool)
        total_reward = jnp.zeros((pop_size * self.num_episodes,))
        counter = 0
        brax_state = brax_reset(keys)
        while counter < self.max_episode_length and ~done.all():
            model_state, action = self._model_forward(
                model_state,
                from_jax_array(brax_state.obs).view(pop_size, self.num_episodes, -1),
            )
            action = action.view(pop_size * self.num_episodes, -1)
            brax_state = brax_step(brax_state, to_jax_array(action))
            done = brax_state.done * (1 - done)
            total_reward += (1 - done) * brax_state.reward
            counter += 1
        # Return
        model_state["key"] = from_jax_array(key)
        total_reward = from_jax_array(total_reward).view(pop_size, self.num_episodes)
        return model_state, total_reward

    # TODO: we will add visualization in the future
    # @torch.jit.ignore
    # def visualize(
    #     self,
    #     key,
    #     weights,
    #     output_type: str = "HTML",
    #     respect_done: bool = False,
    #     max_episode_length: Optional[int] = None,
    #     *args,
    #     **kwargs,
    # ):
    #     """Visualize the brax environment with the given policy and weights.

    #     Parameters
    #     ----------
    #     key
    #         The random key.
    #     weights
    #         The weights of the policy.
    #     output_type
    #         The output type, either "HTML" or "rgb_array".
    #     respect_done
    #         Whether to respect the done signal.
    #     max_episode_length
    #         Used to override the max_episode_length in the constructor.
    #         If None, use the max_episode_length in the constructor.
    #     """
    #     assert output_type in [
    #         "HTML",
    #         "rgb_array",
    #     ], "output_type must be either HTML or rgb_array"

    #     env = envs.get_environment(env_name=self.env_name, backend=self.backend)
    #     brax_state = jax.jit(env.reset)(key)
    #     jit_env_step = jit(env.step)
    #     trajectory = [brax_state.pipeline_state]
    #     episode_length = 1

    #     if self.stateful_policy:
    #         rollout_state = (self.initial_state, brax_state)
    #     else:
    #         rollout_state = (brax_state,)

    #     max_episode_length = max_episode_length or self.max_episode_length
    #     for _ in range(max_episode_length):
    #         if self.stateful_policy:
    #             state, brax_state = rollout_state
    #             action, state = self.policy(state, weights, brax_state.obs)
    #             rollout_state = (state, brax_state)
    #         else:
    #             (brax_state,) = rollout_state
    #             action = self.policy(weights, brax_state.obs)
    #             rollout_state = (brax_state,)

    #         trajectory.append(brax_state.pipeline_state)
    #         brax_state = jit_env_step(brax_state, action)
    #         trajectory.append(brax_state.pipeline_state)
    #         episode_length += 1 - brax_state.done

    #         if respect_done and brax_state.done:
    #             break

    #     if output_type == "HTML":
    #         return html.render(env.sys, trajectory, *args, **kwargs)
    #     else:
    #         return image.render_array(env.sys, trajectory, **kwargs)
