import warnings
from functools import partial
from typing import Callable, Dict, List, Optional, Union

import jax
import jax.numpy as jnp
from jax import jit, lax, pmap, pure_callback
from jax.sharding import PositionalSharding
from jax.tree_util import tree_map

from evox import Algorithm, Problem, State, Stateful, Monitor, jit_method
from evox.utils import parse_opt_direction, algorithm_has_init_ask


class StdWorkflow(Stateful):
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
        monitors: List[Monitor] = [],
        opt_direction: Union[str, List[str]] = "min",
        sol_transforms: List[Callable] = [],
        fit_transforms: List[Callable] = [],
        pop_transform: Optional[Callable] = None,
        jit_problem: bool = True,
        jit_monitor: bool = False,
        num_objectives: Optional[int] = None,
        monitor=None,
    ):
        """
        Parameters
        ----------
        algorithm
            The algorithm.
        problem
            The problem.
        monitor
            Optional monitor(s).
            Configure a single monitor or a list of monitors.
            The monitors will be called in the order of the list.
        opt_direction
            The optimization direction, can be either "min" or "max"
            or a list of "min"/"max" to specific the direction for each objective.
        sol_transform
            Optional candidate solution transform function,
            usually used to decode the candidate solution
            into the format that can be understood by the problem.
            Should be a list of functions,
            and the functions will be applied in the order of the list.
        fit_transforms
            Optional fitness transform function.
            usually used to apply fitness shaping.
            Should be a list of functions,
            and the functions will be applied in the order of the list.
        jit_problem
            If the problem can be jit compiled by JAX or not.
            Default to True.
        num_objectives
            Number of objectives.
            When the problem can be jit compiled, this field is not needed.
            When the problem cannot be jit compiled, this field should be set,
            if not, default to 1.
        """
        self.algorithm = algorithm
        self.problem = problem
        self.monitors = monitors
        if monitor is not None:
            warnings.warn(
                "`monitor` is deprecated, use the `monitors` parameter with a list of monitors instead",
                DeprecationWarning,
            )
            self.monitors = [monitor]
        self.registered_hooks = {
            "pre_step": [],
            "pre_ask": [],
            "post_ask": [],
            "pre_eval": [],
            "post_eval": [],
            "pre_tell": [],
            "post_tell": [],
            "post_step": [],
        }
        for monitor in self.monitors:
            hooks = monitor.hooks()
            for hook in hooks:
                self.registered_hooks[hook].append(monitor)

        self.opt_direction = parse_opt_direction(opt_direction)
        for monitor in self.monitors:
            monitor.set_opt_direction(self.opt_direction)

        self.sol_transforms = sol_transforms
        # for compatibility purpose
        if pop_transform is not None:
            warnings.warn(
                "`pop_transform` is deprecated, use `sol_transforms` with a list of transforms instead",
                DeprecationWarning,
            )
            self.sol_transforms = [pop_transform]
        self.fit_transforms = fit_transforms
        self.jit_problem = jit_problem
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

        # a prototype step function
        # will be then wrapped to get _step
        # We are doing this as a workaround for JAX's static shape requirement
        # Since init_ask and ask can return different shape
        # and jax.lax.cond requires the same shape from two different branches
        # we can only apply lax.cond outside of each `step`
        def _proto_step(self, is_init, state):
            for monitor in self.registered_hooks["pre_ask"]:
                monitor.pre_ask(state)

            if is_init:
                ask = self.algorithm.init_ask
                tell = self.algorithm.init_tell
            else:
                ask = self.algorithm.ask
                tell = self.algorithm.tell

            # candidate solution
            cand_sol, state = ask(state)
            cand_sol_size = cand_sol.shape[0]

            for monitor in self.registered_hooks["post_ask"]:
                monitor.post_ask(state, cand_sol)

            if self.distributed_step is True:
                cand_sol = jax.lax.dynamic_slice_in_dim(
                    cand_sol, state.start_index, self.slice_size, axis=0
                )

            transformed_cand_sol = cand_sol
            for transform in self.sol_transforms:
                transformed_cand_sol = transform(transformed_cand_sol)

            for monitor in self.registered_hooks["pre_eval"]:
                monitor.pre_eval(state, cand_sol, transformed_cand_sol)

            # if the function is jitted
            if self.jit_problem:
                fitness, state = self.problem.evaluate(state, transformed_cand_sol)
            else:
                if self.num_objectives == 1:
                    fit_shape = (cand_sol_size,)
                else:
                    fit_shape = (cand_sol_size, self.num_objectives)
                fitness, state = pure_callback(
                    self.problem.evaluate,
                    (
                        jax.ShapeDtypeStruct(fit_shape, dtype=jnp.float32),
                        state,
                    ),
                    state,
                    transformed_cand_sol,
                )

            if self.distributed_step is True:
                fitness = jax.lax.all_gather(fitness, "node", axis=0, tiled=True)

            fitness = fitness * self.opt_direction

            for monitor in self.registered_hooks["post_eval"]:
                monitor.post_eval(state, cand_sol, transformed_cand_sol, fitness)

            transformed_fitness = fitness
            for transform in self.fit_transforms:
                transformed_fitness = transform(transformed_fitness)

            for monitor in self.registered_hooks["pre_tell"]:
                monitor.pre_tell(
                    state, cand_sol, transformed_cand_sol, fitness, transformed_fitness
                )

            state = tell(state, transformed_fitness)

            for monitor in self.registered_hooks["post_tell"]:
                monitor.post_tell(state)

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

            fitness, new_state = self.problem.evaluate(new_state, pop)

            if self.distributed_step is True:
                fitness = jax.lax.all_gather(fitness, "node", axis=0, tiled=True)

            return fitness, state

        self._valid = jit_method(_valid)

    def setup(self, key):
        return State(generation=0)

    def step(self, state):
        for monitor in self.registered_hooks["pre_step"]:
            monitor.pre_step(state)

        state = self._step(self, state)

        for monitor in self.registered_hooks["post_step"]:
            monitor.post_step(state)

        return state

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
