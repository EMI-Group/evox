import warnings
from functools import partial
from typing import Callable, Dict, List, Optional, Union

import jax
import jax.numpy as jnp
from jax import jit, lax, pmap, pure_callback
from jax.experimental import io_callback
from jax.sharding import PositionalSharding, SingleDeviceSharding
from jax.tree_util import tree_map

from evox import Algorithm, Problem, State, Stateful, jit_method
from evox.utils import parse_opt_direction, algorithm_has_init_ask


class UniWorkflow(Stateful):
    """Experimental unified workflow,
    designed to provide unparallel performance for EC workflow.

    Provide automatic multi-device (e.g. multiple gpus) computation
    as well as distributed computation using JAX's native components.

    Monitor is called using JAX's asynchronous host callback,
    thus closing the monitor is needed to wait for the callback to complete.
    """

    def __init__(
        self,
        algorithm: Algorithm,
        problem: Union[Problem, List[Problem]],
        monitor=None,
        opt_direction: Union[str, List[str]] = "min",
        pop_transform: Optional[Union[Callable, List[Problem]]] = None,
        fit_transform: Optional[Callable] = None,
        record_pop: bool = False,
        record_time: bool = False,
        metrics: Optional[Dict[str, Callable]] = None,
        jit_problem: bool = True,
        jit_monitor: bool = False,
        num_objectives: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        algorithm
            The algorithm.
        problem
            The problem.
        monitor
            Optional monitor.
        opt_direction
            The optimization direction, can be either "min" or "max"
            or a list of "min"/"max" to specific the direction for each objective.
        pop_transform
            Optional population transform function,
            usually used to decode the population
            into the format that can be understood by the problem.
        fit_transform
            Optional fitness transform function.
            usually used to apply fitness shaping.
        record_pop
            Whether to record the population if monitor is enabled.
        record_time
            Whether to record the time at the end of each generation.
            Due to its timing nature,
            record_time requires synchronized functional call.
            Default to False.
        jit_problem
            If the problem can be jit compiled by JAX or not.
            Default to True.
        jit_monitor
            If the monitor can be jit compiled by JAX or not.
            Default to False.
        num_objectives
            Number of objectives.
            When the problem can be jit compiled, this field is not needed.
            When the problem cannot be jit compiled, this field should be set,
            if not, default to 1.
        """
        self.algorithm = algorithm
        self.problem = problem
        self.monitor = monitor

        self.pop_transform = pop_transform
        self.fit_transform = fit_transform
        self.record_pop = record_pop
        self.record_time = record_time
        self.metrics = metrics
        self.jit_problem = jit_problem
        self.jit_monitor = jit_monitor
        self.num_objectives = num_objectives
        self.distributed_step = False
        self.opt_direction = parse_opt_direction(opt_direction)
        self.monitor.set_opt_direction(self.opt_direction)
        if jit_problem is False and self.num_objectives is None:
            warnings.warn(
                (
                    "Using external problem "
                    "but num_objectives isn't set "
                    "assuming to be 1."
                )
            )
            self.num_objectives = 1

        # a prototype step function
        # will be then wrapped to get _step
        # We are doing this as a workaround for JAX's static shape requirement
        # Since init_ask and ask can return different shape
        # and jax.lax.cond requires the same shape from two different branches
        # we can only apply lax.cond outside of each `step`
        def _proto_step(self, is_init, state):
            if is_init:
                ask = self.algorithm.init_ask
                tell = self.algorithm.init_tell
            else:
                ask = self.algorithm.ask
                tell = self.algorithm.tell

            monitor_device = SingleDeviceSharding(jax.devices()[0])
            if self.monitor and self.record_time:
                io_callback(self.monitor.record_time, None, sharding=monitor_device)

            pop, state = ask(state)
            pop_size = pop.shape[0]

            if self.monitor and self.record_pop:
                io_callback(self.monitor.record_pop, None, pop, sharding=monitor_device)

            if self.distributed_step is True:
                pop = jax.lax.dynamic_slice_in_dim(
                    pop, state.start_index, self.slice_size, axis=0
                )

            if self.pop_transform:
                if isinstance(self.pop_transform, list):
                    pop = [transform(pop) for transform in self.pop_transform]
                else:
                    pop = self.pop_transform(pop)

            # if the function is jitted
            if self.jit_problem:
                if isinstance(self.problem, list):
                    fitness = []
                    for decoded in pop:
                        fit, state = self.problem.evaluate(state, decoded)
                        fitness.append(fit)
                    fitness = jnp.concatenate(fitness, axis=0)
                else:
                    fitness, state = self.problem.evaluate(state, pop)
            else:
                if self.num_objectives == 1:
                    fit_shape = (pop_size,)
                else:
                    fit_shape = (pop_size, self.num_objectives)
                fitness, state = pure_callback(
                    self.problem.evaluate,
                    (
                        jax.ShapeDtypeStruct(fit_shape, dtype=jnp.float32),
                        state,
                    ),
                    state,
                    pop,
                )

            if self.distributed_step is True:
                fitness = jax.lax.all_gather(fitness, "node", axis=0, tiled=True)

            fitness = fitness * self.opt_direction

            if self.monitor:
                if self.metrics:
                    metrics = {
                        name: func(fitness) for name, func in self.metrics.items()
                    }
                else:
                    metrics = None
                io_callback(
                    self.monitor.record_fit,
                    None,
                    fitness,
                    metrics,
                    sharding=monitor_device,
                )

            if self.fit_transform:
                fitness = self.fit_transform(fitness)

            state = tell(state, fitness)

            return state.update(generation=state.generation + 1)

        # wrap around _proto_step
        # to handle init_ask and init_tell
        def _step(self, state):
            # probe if self.algorithm has override the init_ask function
            if algorithm_has_init_ask(self.algorithm, state):
                return lax.cond(
                    state.generation == 0,
                    partial(_proto_step, self, True),
                    partial(_proto_step, self, False),
                    state,
                )
            else:
                return _proto_step(self, False, state)

        # the first argument is self, which should be static
        self._step = jit(_step, static_argnums=[0])

        def _valid(self, state, metric):
            new_state = self.problem.valid(state, metric=metric)
            pop, new_state = self.algorithm.ask(new_state)

            if self.distributed_step is True:
                pop = jax.lax.dynamic_slice_in_dim(
                    pop, self.start_index, self.slice_size, axis=0
                )

            if self.pop_transform is not None:
                pop = self.pop_transform(pop)

            fitness, new_state = self.problem.evaluate(new_state, pop)

            if self.distributed_step is True:
                fitness = jax.lax.all_gather(fitness, "node", axis=0, tiled=True)

            return fitness, state

        self._valid = jit_method(_valid)

    def setup(self, key):
        return State(generation=0)

    def step(self, state):
        return self._step(self, state)

    def valid(self, state, metric):
        return self._valid(self, state, metric)

    def _auto_shard(self, state, sharding, pop_size, dim):
        def get_shard_for_array(arr):
            if isinstance(arr, jax.Array):
                if arr.ndim == 2:
                    if arr.shape == (pop_size, dim):
                        return sharding
                    elif arr.shape[0] == pop_size:
                        return sharding.replicate(1)
                    elif arr.shape[1] == dim:
                        return sharding.replicate(0)
                elif arr.ndim == 1 and arr.shape[0] == dim:
                    return sharding.replicate(0, keepdims=False)
                elif arr.ndim == 1 and arr.shape[0] == pop_size:
                    return sharding.replicate(1, keepdims=False)

            return sharding.replicate()

        return tree_map(get_shard_for_array, state)

    def enable_multi_devices(
        self, state: State, devices: Optional[list] = None
    ) -> State:
        """
        Enable the workflow to run on multiple local devices.

        Parameters
        ----------
        state
            The state.
        devices
            A list of devices.
            If set to None, all local devices will be used.

        Returns
        -------
        State
            The sharded state, distributed amoung all devices.
        """
        if self.jit_problem is False:
            raise ValueError(
                "multi-devices with non jit problem isn't currently supported"
            )
        if devices is None:
            devices = jax.local_devices()
        device_count = len(devices)
        dummy_pop, _ = jax.eval_shape(self.algorithm.ask, state)
        pop_size, dim = dummy_pop.shape
        sharding = PositionalSharding(devices).reshape(1, device_count)
        state_sharding = self._auto_shard(state, sharding, pop_size, dim)
        state = jax.device_put(state, state_sharding)
        self._step = jit(
            self._step,
            in_shardings=(state_sharding,),
            out_shardings=state_sharding,
            static_argnums=0,
        )
        return state

    def enable_distributed(self, state):
        """
        Enable the distributed workflow to run across multiple nodes.
        To use jax's distribution ability,
        one need to run the same program on all nodes
        with different parameters in `jax.distributed.initialize`.

        Parameters
        ----------
        state
            The state.

        Returns
        -------
        State
            The sharded state, distributed amoung all nodes.
        """
        # auto determine pop_size and dimension
        dummy_pop, _ = jax.eval_shape(self.algorithm.ask, state)
        pop_size, _dim = dummy_pop.shape
        total_device_count = jax.device_count()
        local_device_count = jax.local_device_count()
        self.slice_size = pop_size // total_device_count
        process_index = jax.process_index()
        start_index = process_index * local_device_count * self.slice_size
        start_indices = [
            start_index + i * self.slice_size for i in range(local_device_count)
        ]
        self.distributed_step = True
        # enter pmap env, thus allowing collective ops in _step and _valid
        self._step = pmap(self._step, axis_name="node", static_broadcasted_argnums=0)
        # pmap requires an extra dimension
        state = jax.device_put_replicated(state, jax.local_devices())
        start_index = jax.device_put_sharded(start_indices, jax.local_devices())
        return state.update(start_index=start_index)
