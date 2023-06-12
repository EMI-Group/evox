from typing import Callable, Optional, Tuple
from functools import partial

import jax
import jax.numpy as jnp
from jax import pmap

from evox import Algorithm, Problem, State, Stateful


class MultiDevicePipeline(Stateful):
    def __init__(
        self,
        algorithm: Algorithm,
        problem: Problem,
        pop_size: int,
        devices: Optional[list] = None,
        pop_transform: Optional[Callable] = None,
        fitness_transform: Optional[Callable] = None,
        global_fitness_transform: Optional[Callable] = None,
    ):
        """Create a multi-device pipeline

        This multi-device pipeline is similar to the DistributedPipeline,
        but use JAX-primitives to implement the distributed framework.
        So compare to DistributedPipeline,
        MultiDevicePipeline has more restrictionis but is more efficient.

        ``pop_transform`` and ``fitness_transform`` are applied at each node,
        while ``global_fitness_transform`` is applied at the main node once per step,
        so monitor should be passed as ``global_fitness_transform``.

        Using MultiDevicePipeline requires
        ``algorithm``, ``problem``, ``pop_transform`` and ``fitness_transform`` to be jit-able.

        Parameters
        ----------
        algorithm
            The algorithm.
        problem
            The problem.
        pop_size
            Population size, this argument together with devices
            will be used to determine the sharding strategy.
        devices
            A list of devices to use.
            Default to None, in which case ``jax.local_devices()`` will be used.
        pop_transform:
            Population transform, this transform is applied at each worker node.
        fitness_transform:
            Fitness transform, this transform is applied at each worker node.
        global_fitness_transform:
            This transform is applied at the main node.
        """
        self._algorithm = algorithm
        self._problem = problem
        self.pop_size = pop_size

        if devices is None:
            self.devices = jax.local_devices()
        else:
            self.devices = devices

        self.num_devices = len(self.devices)

        assert pop_size % self.num_devices == 0
        self.slice_size = pop_size // self.num_devices
        self.start_indices = jax.device_put_sharded(
            list(range(0, self.pop_size, self.slice_size)), self.devices
        )

        self.pop_transform = pop_transform
        if fitness_transform is None:
            self.fitness_transform = None
        else:
            self.fitness_transform = pmap(fitness_transform)
        self.global_fitness_transform = global_fitness_transform

        # cache pmap result
        self.pmap_alg_ask = pmap(self._algorithm.ask)
        self.pmap_alg_tell = pmap(self._algorithm.tell)
        self.pmap_pro_eval = pmap(self._problem.evaluate)
        self.pmap_slice = pmap(self._get_one_slice)

    def setup(self, key: jax.Array) -> State:
        alg_state = self._algorithm.init(key)
        alg_states = jax.device_put_replicated(alg_state, self.devices)

        pro_state = self._problem.init(key)
        pro_states = jax.device_put_replicated(pro_state, self.devices)

        return State(alg_states=alg_states, pro_states=pro_states)

    def _get_one_slice(self, pop: jax.Array, start_index: int) -> jax.Array:
        return jax.lax.dynamic_slice_in_dim(pop, start_index, self.slice_size, axis=0)

    def _slice_population(self, pop: jax.Array) -> jax.Array:
        return self.pmap_slice(pop, self.start_indices)

    def step(self, state: State) -> State:
        pop, alg_states = self.pmap_alg_ask(state.alg_states)
        pop = self._slice_population(pop)
        if self.pop_transform is not None:
            pop = pmap(self.pop_transform)(pop)

        fitness, pro_states = self.pmap_pro_eval(state.pro_states, pop)

        if self.fitness_transform is not None:
            fitness = self.fitness_transform(fitness)

        fitness = jnp.concatenate(fitness, axis=0)
        if self.global_fitness_transform is not None:
            fitness = self.global_fitness_transform(fitness)

        fitness = jax.device_put_replicated(fitness, self.devices)

        alg_states = self.pmap_alg_tell(alg_states, fitness)

        return state.update(
            alg_states=alg_states,
            pro_states=pro_states,
        )

    def valid(self, state: State, metric: str = "loss") -> Tuple[jax.Array, State]:
        new_pro_state = pmap(partial(self.problem.valid, metric=metric))(
            state.pro_states
        )
        pop, new_state = self.pmap_alg_ask(new_state)
        if self.pop_transform is not None:
            pop = self.pop_transform(pop)

        new_state, fitness = self.pmap_pro_eval(new_pro_state, pop)
        return fitness, state

    def sample(self, state: State):
        """Sample the algorithm but don't change it's state"""
        sample_pop, state_ = self.pmap_alg_ask(state)
        if self.pop_transform is not None:
            sample_pop = self.pop_transform(sample_pop)

        # we have N copies of the population
        # just take one
        sample_pop = sample_pop[0]

        return sample_pop, state
