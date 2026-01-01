from collections.abc import Callable, Sequence
import dataclasses
from functools import partial
from typing import NamedTuple, Optional, Union
import warnings

import jax
import jax.experimental
import jax.experimental.multihost_utils
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import lax

from evox import (
    Algorithm,
    Monitor,
    Problem,
    State,
    Workflow,
    dataclass,
    has_init_ask,
    has_init_tell,
    pytree_field,
    use_state,
)
from evox.core.distributed import all_gather, POP_AXIS_NAME, ShardingType
from evox.utils import parse_opt_direction


def _leftover_callbacks_warning(method_name):
    warnings.warn(
        f"`{method_name}` is called with a state that has leftover callbacks. "
        "Did you forget to call `execute_callbacks`?"
    )


@dataclass
class StdWorkflowState:
    generation: int
    first_step: bool = pytree_field(static=True)


class MultiDeviceConfig(NamedTuple):
    devices: list[jax.Device]
    axis_name: str


@dataclass
class StdWorkflow(Workflow):
    """Experimental unified workflow,
    designed to provide unparallel performance for EC workflow.

    Provide automatic multi-device (e.g. multiple gpus) computation
    as well as distributed computation using JAX's native components.

    Monitor is called using JAX's asynchronous host callback,
    thus closing

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
    solution_transforms
        Optional candidate solution transform function,
        usually used to decode the candidate solutions
        into the format that can be understood by the problem.
        Should be a list of functions,
        and the functions will be applied in the order of the list.
        Each function should have the signature :code:`fn(solutions) -> solutions`,
        where solutions outputed by the EC algorithm.
    fitness_transforms
        Optional fitness transform function.
        usually used to apply fitness shaping.
        Should be a list of functions,
        and the functions will be applied in the order of the list.
        Each function should have the signature :code:`fn(fitness) -> fitness`,
        where fitness outputed by the problem.
    jit_step:
        Whether jit the entire step function.
        Default to True
    external_problem
        Tell workflow whether the problem is external that cannot be jitted.
        Default to False.
    auto_exec_callbacks
        Whether to automatically call `execute_callbacks` on the state at the end of each step.
        Default to True.
    clear_monitor_history
        Whether to clear the monitor history at the beginning of the workflow.
        Default to True.
    num_objectives
        Number of objectives. Used when external_problem=True.
        When the problem cannot be jitted, JAX cannot infer the shape, and
        this field should be manually set. the monitor is needed to wait for the callback to complete.
    """

    algorithm: Algorithm
    problem: Problem
    monitors: Sequence[Monitor] = pytree_field(default=(), metadata={"nested": True})
    opt_direction: Union[str, Sequence[str]] = pytree_field(default="min", static=True)
    solution_transforms: Sequence[Callable[[jax.Array], jax.Array]] = pytree_field(
        default=(), static=True
    )
    fitness_transforms: Sequence[Callable[[jax.Array], jax.Array]] = pytree_field(
        default=(), static=True
    )
    jit_step: bool = pytree_field(default=True, static=True)
    external_problem: bool = pytree_field(default=False, static=True)
    auto_exec_callbacks: bool = pytree_field(default=True, static=True)
    clear_monitor_history: bool = pytree_field(default=True, static=True)
    num_objectives: Optional[int] = pytree_field(default=None, static=True)
    multi_device_config: Optional[MultiDeviceConfig] = pytree_field(
        default=None, static=True
    )
    migrate_helper: Optional[Callable] = pytree_field(default=None, static=True)

    # inner
    _step: Callable[[State], State] = pytree_field(static=True, init=False)
    _registered_hooks: dict = pytree_field(static=True, init=False)
    _pmap_axis_name: str = pytree_field(static=True, init=False)
    _opt_direction_mask: jnp.array = pytree_field(init=False)

    def __post_init__(self):
        if self.external_problem is True and self.num_objectives is None:
            raise ValueError(("Using external problem, but num_objectives isn't set "))

        registered_hooks = {
            "pre_step": [],
            "pre_ask": [],
            "post_ask": [],
            "pre_eval": [],
            "post_eval": [],
            "pre_tell": [],
            "post_tell": [],
            "post_step": [],
        }
        for i, monitor in enumerate(self.monitors):
            hooks = monitor.hooks()
            for hook in hooks:
                registered_hooks[hook].append(monitor)
        self.set_frozen_attr("_registered_hooks", registered_hooks)

        opt_direction = parse_opt_direction(self.opt_direction)
        for monitor in self.monitors:
            monitor.set_opt_direction(opt_direction)
        self.set_frozen_attr("_opt_direction_mask", opt_direction)

        def _step(self, state):
            state = self._pre_step_hook(state)
            state = self._pre_ask_hook(state)
            cands, state = self._ask(state)

            if self.multi_device_config:
                # when using multi devices
                # force the candidates to be sharded along the first axis
                cands = jax.lax.with_sharding_constraint(
                    cands,
                    ShardingType.SHARED_FIRST_DIM.get_sharding(
                        self.multi_device_config.devices
                    ),
                )

            state = self._post_ask_hook(state, cands)

            transformed_cands = cands
            for transform in self.solution_transforms:
                transformed_cands = transform(transformed_cands)

            state = self._pre_eval_hook(state, transformed_cands)
            fitness, state = self._evaluate(state, transformed_cands)

            if self.multi_device_config:
                # when using multi devices
                # force the fitness to be replicated
                fitness = jax.lax.with_sharding_constraint(
                    fitness,
                    ShardingType.REPLICATED.get_sharding(
                        self.multi_device_config.devices
                    ),
                )

            state = self._post_eval_hook(state, fitness)

            transformed_fitness = fitness
            for transform in self.fitness_transforms:
                transformed_fitness = transform(transformed_fitness)

            state = self._pre_tell_hook(state, transformed_fitness)
            state = self._tell(state, transformed_fitness)
            state = self._post_tell_hook(state)

            if self.migrate_helper is not None:
                do_migrate, foreign_populations, foreign_fitness = (
                    self.migrate_helper.migrate_from_human()
                )

                state = lax.cond(
                    do_migrate,
                    lambda state, pop, fit: use_state(self.algorithm.migrate)(
                        state, pop, fit
                    ),
                    lambda state, _pop, _fit: state,
                    state,
                    foreign_populations,
                    foreign_fitness,
                )

            if has_init_ask(self.algorithm) and state.first_step:
                # this ensures that _step() will be re-jitted
                state = state.replace(generation=state.generation + 1, first_step=False)
            else:
                state = state.replace(generation=state.generation + 1)

            state = self._post_step_hook(state)

            return state

        if self.jit_step:
            # the first argument is self, which should be static
            if dataclasses.is_dataclass(self.algorithm) and dataclasses.is_dataclass(
                self.problem
            ):
                _step = jax.jit(_step)
            else:
                _step = jax.jit(_step, static_argnums=(0,))

        self.set_frozen_attr("_step", _step)
        self.set_frozen_attr("_pmap_axis_name", None)

        if self.clear_monitor_history:
            for monitor in self.monitors:
                monitor.clear_history()

    def _ask(self, state):
        if has_init_ask(self.algorithm) and state.first_step:
            ask = self.algorithm.init_ask
        else:
            ask = self.algorithm.ask

        # candidate: individuals that need to be evaluated (may differ from population)
        # Note: num_cands can be different from init_ask() and ask()
        cands, state = use_state(ask)(state)

        return cands, state

    def _evaluate(self, state, transformed_cands):
        num_cands = jtu.tree_leaves(transformed_cands)[0].shape[0]

        # if the function is jitted
        if not self.external_problem:
            fitness, state = use_state(self.problem.evaluate)(state, transformed_cands)
        else:
            if self.num_objectives == 1:
                fit_shape = (num_cands,)
            else:
                fit_shape = (num_cands, self.num_objectives)
            fitness, state = jax.pure_callback(
                use_state(self.problem.evaluate),
                (
                    jax.ShapeDtypeStruct(fit_shape, dtype=jnp.float32),
                    state,
                ),
                state,
                transformed_cands,
            )

        fitness = all_gather(fitness, self._pmap_axis_name, axis=0, tiled=True)
        fitness = fitness * self._opt_direction_mask

        return fitness, state

    def _tell(self, state, transformed_fitness):
        if has_init_tell(self.algorithm) and state.first_step:
            tell = self.algorithm.init_tell
        else:
            tell = self.algorithm.tell

        state = use_state(tell)(state, transformed_fitness)

        return state

    def _pre_step_hook(self, state):
        for monitor in self._registered_hooks["pre_step"]:
            state = use_state(monitor.pre_step)(state, state)
        return state

    def _pre_ask_hook(self, state):
        for monitor in self._registered_hooks["pre_ask"]:
            state = use_state(monitor.pre_ask)(state, state)
        return state

    def _post_ask_hook(self, state, cands):
        for monitor in self._registered_hooks["post_ask"]:
            state = use_state(monitor.post_ask)(state, state, cands)
        return state

    def _pre_eval_hook(self, state, transformed_cands):
        for monitor in self._registered_hooks["pre_eval"]:
            state = use_state(monitor.pre_eval)(state, state, transformed_cands)
        return state

    def _post_eval_hook(self, state, fitness):
        for monitor in self._registered_hooks["post_eval"]:
            state = use_state(monitor.post_eval)(state, state, fitness)
        return state

    def _pre_tell_hook(self, state, transformed_fitness):
        for monitor in self._registered_hooks["pre_tell"]:
            state = use_state(monitor.pre_tell)(state, state, transformed_fitness)
        return state

    def _post_tell_hook(self, state):
        for monitor in self._registered_hooks["post_tell"]:
            state = use_state(monitor.post_tell)(state, state)
        return state

    def _post_step_hook(self, state):
        for monitor in self._registered_hooks["post_step"]:
            state = use_state(monitor.post_step)(state, state)
        return state

    def setup(self, key):
        return StdWorkflowState(generation=0, first_step=True)

    def step(self, state):
        if self.auto_exec_callbacks and state._callbacks:
            _leftover_callbacks_warning("step")

        state = self._step(self, state)

        if self.auto_exec_callbacks:
            state = state.execute_callbacks(state)
        return state

    def enable_multi_devices(
        self,
        state: State,
        devices: Optional[list[jax.Device]] = None,
    ) -> State:
        """
        Enable the workflow to run on multiple devices.
        Multiple nodes(processes) are also supported.
        To specify which devices are used, use env vars like `CUDA_VISIBLE_DEVICES`

        Parameters
        ----------
        state
            The state.
        devices
            The devices to use.
            If None, then by default the function will use all local devices (`jax.local_devices()`).
            Otherwise, specify a list of jax.Device.

        Returns
        -------
        State
            The sharded state, distributed amoung all devices
            with additional distributed information.

        Examples
        --------
        >>> state = workflow.enable_multi_devices(state)
        >>> for i in range(100):
        ...     state = workflow.step(state) # now it runs on multiple devices
        """
        if not devices:
            # auto select all local devices
            devices = jax.local_devices()

        self.multi_device_config = MultiDeviceConfig(
            devices=devices,
            axis_name=POP_AXIS_NAME,
        )

        sharding = state.get_sharding(devices)
        state = jax.device_put(state, sharding)
        self.set_frozen_attr("_pmap_axis_name", POP_AXIS_NAME)

        return state
