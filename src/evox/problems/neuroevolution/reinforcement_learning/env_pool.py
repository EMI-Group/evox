from typing import Callable, Optional

import envpool
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, random, vmap, lax
from jax.experimental import io_callback

from evox import Problem, State, jit_class
from evox.utils.io import to_x32_if_needed, x32_func_call


@jit_class
class EnvPool(Problem):
    def __init__(
        self,
        policy: Callable,
        env_name: str,
        num_envs: int,
        env_options: dict = {},
        cap_episode_length: Optional[int] = None,
    ):
        self.batch_policy = jit(vmap(policy))
        self.num_envs = num_envs
        self.env = envpool.make(
            env_name,
            num_envs=num_envs,
            env_type="gymnasium",
            **env_options,
        )
        self.cap_episode_length = cap_episode_length

    def setup(self, key):
        return State(key=key)

    def evaluate(self, state, pop):
        key, subkey = random.split(state.key)
        seed = random.randint(subkey, (1,), 0, jnp.iinfo(jnp.int32).max)
        io_callback(self.env.seed, None, seed)
        obs, info = to_x32_if_needed(self.env.reset(None))
        obs, info = io_callback(x32_func_call(self.env.reset), (obs, info), None)
        total_reward = 0
        i = 0

        def cond_func(loop_state):
            i, done, _total_reward, _obs = loop_state
            if self.cap_episode_length:
                return (i < self.cap_episode_length) & ~jnp.all(done)
            else:
                return ~jnp.all(done)

        def step(loop_state):
            i, done, total_reward, obs = loop_state
            action = self.batch_policy(pop, obs)
            obs, reward, terminated, truncated, info = to_x32_if_needed(
                self.env.step(np.zeros(action.shape))
            )
            obs, reward, terminated, truncated, info = io_callback(
                x32_func_call(lambda action: self.env.step(np.copy(action))),
                (obs, reward, terminated, truncated, info),
                action,
            )
            total_reward += ~done * reward
            done = done | (terminated | truncated)

            return i + 1, done, total_reward, obs

        _i, _term, total_reward, _obs = lax.while_loop(
            cond_func,
            step,
            (
                0,
                jnp.zeros(self.num_envs, dtype=jnp.bool_),
                jnp.zeros(self.num_envs),
                obs,
            ),
        )

        return total_reward, state.update(key=key)
