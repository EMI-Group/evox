__all__ = ["BraxProblem"]

import copy
import weakref
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import torch
import torch.nn as nn
import torch.utils.dlpack
from brax import envs
from brax.io import html, image
from torch._C._functorch import get_unwrapped, is_batchedtensor

from evox.core import Problem, use_state
from evox.utils import VmapInfo

from .utils import get_vmap_model_state_forward


# to_dlpack is not necessary for torch.Tensor and jax.Array
# because they have a __dlpack__ method, which is called by their respective from_dlpack methods.
def to_jax_array(x: torch.Tensor) -> jax.Array:
    # When the torch has GPU support but the jax does not, we need to move the tensor to CPU first.
    if is_batchedtensor(x):
        x = get_unwrapped(x)
    if x.device.type != "cpu" and jax.default_backend() == "cpu":
        return jax.dlpack.from_dlpack(x.detach().cpu())
    return jax.dlpack.from_dlpack(x.detach())


def from_jax_array(x: jax.Array, device: Optional[torch.device] = None) -> torch.Tensor:
    if device is None:
        device = torch.get_default_device()
    return torch.utils.dlpack.from_dlpack(x).to(device)


__brax_data__: Dict[
    int,
    Tuple[
        Callable[[jax.Array], envs.State],  # vmap_brax_reset
        Callable[[envs.State, jax.Array], envs.State],  # vmap_brax_step
        Callable[
            [Dict[str, torch.Tensor], torch.Tensor], Tuple[Dict[str, torch.Tensor], torch.Tensor]
        ],  # vmap_state_forward
        List[str],  # state_keys
    ],
] = {}


def _evaluate_brax_main(
    env_id: int,
    pop_size: int,
    rotate_key: bool,
    num_episodes: int,
    max_episode_length: int,
    key: torch.Tensor,
    model_state: List[torch.Tensor],
) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
    # check the pop_size in the inputs
    # Take a parameter and check its size
    actual_pop_size = model_state[0].size(0)
    assert actual_pop_size == pop_size, (
        f"The actual population size must match the pop_size parameter when creating BraxProblem. Expected: {pop_size}, Actual: {actual_pop_size}"
    )
    device = model_state[0].device
    vmap_brax_reset, vmap_brax_step, vmap_state_forward, state_keys = __brax_data__.get(env_id)
    model_state = {k: v.clone() for k, v in zip(state_keys, model_state)}

    key = to_jax_array(key)
    # For each episode, we need a different random key.
    # For each individual in the population, we need the same set of keys.
    # Loop until environment stops
    if rotate_key:
        key, eval_key = jax.random.split(key)
    else:
        key, eval_key = key, key

    keys = jax.random.split(eval_key, num_episodes)
    keys = jnp.broadcast_to(keys, (pop_size, *keys.shape)).reshape(pop_size * num_episodes, -1)
    done = jnp.zeros((pop_size * num_episodes,), dtype=bool)
    total_reward = jnp.zeros((pop_size * num_episodes,))
    counter = 0
    brax_state = vmap_brax_reset(keys)

    while counter < max_episode_length and ~done.all():
        model_state, action = vmap_state_forward(
            model_state, from_jax_array(brax_state.obs, device).view(pop_size, num_episodes, -1)
        )
        action = action.view(pop_size * num_episodes, -1)
        brax_state = vmap_brax_step(brax_state, to_jax_array(action))
        done = brax_state.done * (1 - done)
        total_reward += (1 - done) * brax_state.reward
        counter += 1

    # Return
    new_key = from_jax_array(key, device)
    total_reward = from_jax_array(total_reward, device)
    total_reward = total_reward.view(pop_size, num_episodes)
    model_state = [model_state[k] for k in state_keys]
    return new_key, model_state, total_reward


@torch.library.custom_op("evox::_evaluate_brax", mutates_args=())
def _evaluate_brax(
    env_id: int,
    pop_size: int,
    rotate_key: bool,
    num_episodes: int,
    max_episode_length: int,
    key: torch.Tensor,
    model_state: List[torch.Tensor],
) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
    return _evaluate_brax_main(env_id, pop_size, rotate_key, num_episodes, max_episode_length, key, model_state)


@_evaluate_brax.register_fake
def _fake_evaluate_brax(
    env_id: int,
    pop_size: int,
    rotate_key: bool,
    num_episodes: int,
    max_episode_length: int,
    key: torch.Tensor,
    model_state: List[torch.Tensor],
) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
    return (
        key.new_empty(key.size()),
        [v.new_empty(v.size()) for v in model_state],
        model_state[0].new_empty(pop_size, num_episodes),
    )


@torch.library.custom_op("evox::_evaluate_brax_vmap_main", mutates_args=())
def _evaluate_brax_vmap_main(
    batch_size: int,
    in_dim: List[int],
    env_id: int,
    pop_size: int,
    rotate_key: bool,
    num_episodes: int,
    max_episode_length: int,
    key: torch.Tensor,
    model_state: List[torch.Tensor],
) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
    # flatten vmap dim and pop dim
    model_state = [(v if d is None else v.movedim(d, 0).flatten(0, 1)) for d, v in zip(in_dim, model_state)]
    key, model_state, reward = _evaluate_brax_main(
        env_id, pop_size, rotate_key, num_episodes, max_episode_length, key, model_state
    )
    model_state = [(v if d is None else v.unflatten(0, (batch_size, -1))) for d, v in zip(in_dim, model_state)]
    reward = reward.unflatten(0, (batch_size, -1))
    return key, model_state, reward


@_evaluate_brax.register_vmap
def _evaluate_brax_vmap(
    vmap_info: VmapInfo,
    in_dims: Tuple[int | None | List[int], ...],
    env_id: int,
    pop_size: int,
    rotate_key: bool,
    num_episodes: int,
    max_episode_length: int,
    key: torch.Tensor,
    model_state: List[torch.Tensor],
) -> Tuple[Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor], Tuple[int | None, List[int], int]]:
    assert all(d is None for d in in_dims[:-1]), "Cannot vmap over `BraxProblem` itself"
    assert in_dims[-1] is not None, "Cannot vmap none of the dimensions"
    key, model_state, reward = _evaluate_brax_vmap_main(
        vmap_info.batch_size,
        in_dims[-1],
        env_id,
        pop_size,
        rotate_key,
        num_episodes,
        max_episode_length,
        key,
        model_state,
    )
    return (key, model_state, reward), (None, [0] * len(model_state), 0)


@_evaluate_brax_vmap_main.register_fake
def _fake_evaluate_brax_vmap(
    batch_size: int,
    in_dim: List[int],
    env_id: int,
    pop_size: int,
    rotate_key: bool,
    num_episodes: int,
    max_episode_length: int,
    key: torch.Tensor,
    model_state: List[torch.Tensor],
) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
    return (
        key.new_empty(key.size()),
        [v.new_empty(v.size()).movedim(d, 0) for d, v in zip(in_dim, model_state)],
        model_state[0].new_empty(batch_size, pop_size // batch_size, num_episodes),
    )


class BraxProblem(Problem):
    """The Brax problem wrapper."""

    def __init__(
        self,
        policy: nn.Module,
        env_name: str,
        max_episode_length: int,
        num_episodes: int,
        seed: int = None,
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
        :param seed: The seed used to create a PRNGKey for the brax environment. When None, randomly select one. Default to None.
        :param pop_size: The size of the population to be evaluated. If None, we expect the input to have a population size of 1.
        :param rotate_key: Indicates whether to rotate the random key for each iteration (default is True). <br/> If True, the random key will rotate after each iteration, resulting in non-deterministic and potentially noisy fitness evaluations. This means that identical policy weights may yield different fitness values across iterations. <br/> If False, the random key remains the same for all iterations, ensuring consistent fitness evaluations.
        :param reduce_fn: The function to reduce the rewards of multiple episodes. Default to `torch.mean`.
        :param backend: Brax's backend. If None, the default backend of the environment will be used. Default to None.
        :param device: The device to run the computations on. Defaults to the current default device.

        ## Notice
        The initial key is obtained from `torch.random.get_rng_state()`.

        ## Warning
        This problem does NOT support HPO wrapper (`problems.hpo_wrapper.HPOProblemWrapper`) out-of-box, i.e., the workflow containing this problem CANNOT be vmapped.
        *However*, by setting `pop_size` to the multiplication of inner population size and outer population size, you can still use this problem in a HPO workflow.
        Yet, the `num_repeats` of HPO wrapper *must* be set to 1, please use the parameter `num_episodes` instead.

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
        vmap_env = envs.wrappers.training.VmapWrapper(env)
        # Compile Brax environment
        self.brax_reset = jax.jit(env.reset)
        self.brax_step = jax.jit(env.step)
        self.vmap_brax_reset = jax.jit(vmap_env.reset)
        self.vmap_brax_step = jax.jit(vmap_env.step)
        # JIT stateful model forward
        self.vmap_init_state, self.vmap_state_forward = get_vmap_model_state_forward(
            model=policy,
            pop_size=pop_size,
            in_dims=(0, 0),
            device=device,
        )
        self.state_forward = torch.compile(use_state(policy))
        if seed is None:
            seed = torch.randint(0, 2**31, (1,)).item()
        self.key = from_jax_array(jax.random.PRNGKey(seed), device)
        copied_policy = copy.deepcopy(policy).to(device)
        self.init_state = copied_policy.state_dict()
        for _name, value in self.init_state.items():
            value.requires_grad = False
        # Store to global
        self.state_keys = list(self.init_state.keys())
        global __brax_data__
        __brax_data__[id(self)] = (
            self.vmap_brax_reset,
            self.vmap_brax_step,
            self.vmap_state_forward,
            self.state_keys,
        )
        weakref.finalize(self, __brax_data__.pop, id(self), None)
        # Store variables
        self._id_ = id(self)
        self.reduce_fn = reduce_fn
        self.rotate_key = rotate_key
        self.pop_size = pop_size
        self.num_episodes = num_episodes
        self.max_episode_length = max_episode_length
        self.env_sys = env.sys
        self.device = device

    # disable torch.compile for JAX code
    @torch.compiler.disable
    def _evaluate_brax_record(
        self,
        model_state: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, List[Any]]:
        key = to_jax_array(self.key)
        # For each episode, we need a different random key.
        # For each individual in the population, we need the same set of keys.
        # Loop until environment stops
        if self.rotate_key:
            key, eval_key = jax.random.split(key)
        else:
            key, eval_key = key, key

        keys = eval_key
        done = jnp.zeros((), dtype=bool)
        total_reward = jnp.zeros(())
        counter = 0
        brax_state = self.brax_reset(keys)
        trajectory = [brax_state.pipeline_state]

        while counter < self.max_episode_length and ~done.all():
            model_state, action = self.state_forward(model_state, from_jax_array(brax_state.obs, self.device))
            brax_state = self.brax_step(brax_state, to_jax_array(action))
            done = brax_state.done * (1 - done)
            total_reward += (1 - done) * brax_state.reward
            counter += 1
            trajectory.append(brax_state.pipeline_state)
        # Return
        self.key = from_jax_array(key, self.device)
        total_reward = from_jax_array(total_reward, self.device)
        return model_state, total_reward, trajectory

    def evaluate(self, pop_params: Dict[str, nn.Parameter]) -> torch.Tensor:
        """Evaluate the final rewards of a population (batch) of model parameters.

        :param pop_params: A dictionary of parameters where each key is a parameter name and each value is a tensor of shape (batch_size, *param_shape) representing the batched parameters of batched models.

        :return: A tensor of shape (batch_size,) containing the reward of each sample in the population.
        """
        # Merge the given parameters into the initial parameters
        model_state = {**self.vmap_init_state, **pop_params}
        # CANNOT COMPILE: model_state = self.vmap_init_state | pop_params
        model_state = [model_state[k] for k in self.state_keys]
        # Brax environment evaluation
        key, _, rewards = _evaluate_brax(
            env_id=self._id_,
            pop_size=self.pop_size,
            rotate_key=self.rotate_key,
            num_episodes=self.num_episodes,
            max_episode_length=self.max_episode_length,
            key=self.key,
            model_state=model_state,
        )
        self.key = key
        rewards = self.reduce_fn(rewards, dim=-1)
        return rewards

    def visualize(
        self,
        weights: Dict[str, nn.Parameter],
        seed: int = 0,
        output_type: str = "HTML",
        *args,
        **kwargs,
    ) -> str | torch.Tensor:
        """Visualize the brax environment with the given policy and weights.

        :param weights: The weights of the policy model. Which is a dictionary of parameters.
        :param output_type: The output type of the visualization, "HTML" or "rgb_array". Default to "HTML".

        :return: The visualization output.
        """
        assert output_type in [
            "HTML",
            "rgb_array",
        ], "output_type must be either HTML or rgb_array"
        model_state = self.init_state | weights
        # Brax environment evaluation
        model_state, _rewards, trajectory = self._evaluate_brax_record(model_state)
        trajectory = [brax_state for brax_state in trajectory]
        if output_type == "HTML":
            return html.render(self.env_sys, trajectory, *args, **kwargs)
        else:
            return image.render_array(self.env_sys, trajectory, **kwargs)
