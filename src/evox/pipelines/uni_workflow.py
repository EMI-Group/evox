from evox import jit_method, Stateful, Algorithm, Problem, State
from typing import Optional, Callable, Dict
import warnings
import jax
import jax.numpy as jnp
from jax import jit, pmap
from jax.tree_util import tree_map
from jax.sharding import PositionalSharding
import jax.experimental.host_callback as hcb


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
        problem: Problem,
        monitor=None,
        pop_transform: Optional[Callable] = None,
        fit_transform: Optional[Callable] = None,
        record_pop: bool = True,
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
        if jit_problem is False and self.num_objectives is None:
            warnings.warn(
                (
                    "Using external problem "
                    "but num_objectives isn't set "
                    "assuming to be 1."
                )
            )
            self.num_objectives = 1

        def _step(self, state):
            if self.monitor and self.record_time:
                hcb.call(self.monitor.record_time, None)

            pop, state = self.algorithm.ask(state)
            pop_size = pop.shape[0]

            if self.monitor and self.record_pop:
                hcb.id_tap(self.monitor.record_pop, pop)

            if self.distributed_step is True:
                pop = jax.lax.dynamic_slice_in_dim(
                    pop, self.start_index, self.slice_size, axis=0
                )

            if self.pop_transform:
                pop = self.pop_transform(pop)

            # if the function is jitted
            if self.jit_problem:
                fitness, state = self.problem.evaluate(state, pop)
            else:
                if self.num_objectives == 1:
                    fit_shape = (pop_size,)
                else:
                    fit_shape = (pop_size, self.num_objectives)
                fitness, state = hcb.call(
                    lambda args: self.problem.evaluate(args[0], args[1]),
                    (state, pop),
                    result_shape=(
                        jax.ShapeDtypeStruct(fit_shape, dtype=jnp.float32),
                        state,
                    ),
                )
            if self.distributed_step is True:
                fitness = jax.lax.all_gather(fitness, "node", axis=0, tiled=True)

            if self.monitor:
                if self.metrics:
                    metrics = {
                        name: func(fitness) for name, func in self.metrics.items()
                    }
                else:
                    metrics = None
                hcb.id_tap(
                    lambda args, _: self.monitor.record_fit(args[0], args[1]),
                    (fitness, metrics),
                )

            if self.fit_transform:
                fitness = self.fit_transform(fitness)

            state = self.algorithm.tell(state, fitness)

            return state

        self._step = jit_method(_step)

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
            static_argnums=[0],
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
        self.slice_size = pop_size // jax.process_count()
        self.start_index = self.slice_size * jax.process_index()
        self.distributed_step = True
        # enter pmap env, thus allowing collective ops in _step and _valid
        self._step = pmap(self._step, axis_name="node", static_broadcasted_argnums=0)
        self._valid = pmap(self._valid, axis_name="node", static_broadcasted_argnums=0, in_axes=(None, 0, None))
        # pmap requires an extra dimension
        state = tree_map(lambda x: jnp.expand_dims(x, axis=0), state)
        return state
