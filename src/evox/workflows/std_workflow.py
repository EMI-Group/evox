from typing import Callable, List, Optional, Union

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from evox import (
    Algorithm,
    Problem,
    State,
    Workflow,
    Monitor,
    use_state,
    dataclass,
    pytree_field,
    has_init_ask,
    has_init_tell,
)
from evox.core.distributed import POP_AXIS_NAME, all_gather, get_process_id
from evox.utils import parse_opt_direction


@dataclass
class StdWorkflowState:
    generation: int
    first_step: bool = pytree_field(static=True)
    rank: int
    world_size: int = pytree_field(static=True)


class StdWorkflow(Workflow):
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
        candidate_transforms: List[Callable] = [],
        fitness_transforms: List[Callable] = [],
        jit_step: bool = True,
        external_problem: bool = False,
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
            Optional monitor(s).
            Configure a single monitor or a list of monitors.
            The monitors will be called in the order of the list.
        opt_direction
            The optimization direction, can be either "min" or "max"
            or a list of "min"/"max" to specific the direction for each objective.
        candidate_transforms
            Optional candidate solution transform function,
            usually used to decode the candidate solution
            into the format that can be understood by the problem.
            Should be a list of functions,
            and the functions will be applied in the order of the list.
        fitness_transforms
            Optional fitness transform function.
            usually used to apply fitness shaping.
            Should be a list of functions,
            and the functions will be applied in the order of the list.
        jit_step:
            Whether jit the entire step function.
            Default to True
        external_problem
            Tell workflow whether the problem is external that cannot be jitted.
            Default to False.
        num_objectives
            Number of objectives. Used when external_problem=True.
            When the problem cannot be jitted, JAX cannot infer the shape, and
            this field should be manually set.
        """
        self.algorithm = algorithm
        self.problem = problem
        self.monitors = monitors

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

        self.candidate_transforms = candidate_transforms
        self.fitness_transforms = fitness_transforms
        self.jit_step = jit_step
        self.external_problem = external_problem
        self.num_objectives = num_objectives
        if self.external_problem is True and self.num_objectives is None:
            raise ValueError(
                ("Using external problem, but num_objectives isn't set ")
            )

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
                fitness, state = use_state(self.problem.evaluate)(
                    state, transformed_cands
                )
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

            fitness = all_gather(fitness, self.pmap_axis_name, axis=0, tiled=True)
            fitness = fitness * self.opt_direction

            return fitness, state

        def _tell(self, state, transformed_fitness):
            if has_init_tell(self.algorithm) and state.first_step:
                tell = self.algorithm.init_tell
            else:
                tell = self.algorithm.tell

            state = use_state(tell)(state, transformed_fitness)

            return state

        def _step(self, state):
            for monitor in self.registered_hooks["pre_step"]:
                monitor.pre_step(state)

            for monitor in self.registered_hooks["pre_ask"]:
                monitor.pre_ask(state)

            cands, state = _ask(self, state)

            for monitor in self.registered_hooks["post_ask"]:
                monitor.post_ask(state, cands)

            num_cands = jtu.tree_leaves(cands)[0].shape[0]
            # in multi-device|host mode, each device only evaluates a slice of the population
            if num_cands % state.world_size != 0:
                raise ValueError(
                    f"#Candidates ({num_cands}) should be divisible by the number of devices ({state.world_size})"
                )
            # Note: slice_size is static
            slice_size = num_cands // state.world_size
            cands = jtu.tree_map(
                lambda x: jax.lax.dynamic_slice_in_dim(
                    x, state.rank * slice_size, slice_size, axis=0
                ),
                cands,
            )

            transformed_cands = cands
            for transform in self.candidate_transforms:
                transformed_cands = transform(transformed_cands)

            for monitor in self.registered_hooks["pre_eval"]:
                monitor.pre_eval(state, cands, transformed_cands)

            fitness, state = _evaluate(self, state, transformed_cands)

            for monitor in self.registered_hooks["post_eval"]:
                monitor.post_eval(state, cands, transformed_cands, fitness)

            transformed_fitness = fitness
            for transform in self.fitness_transforms:
                transformed_fitness = transform(transformed_fitness)

            for monitor in self.registered_hooks["pre_tell"]:
                monitor.pre_tell(
                    state, cands, transformed_cands, fitness, transformed_fitness
                )

            state = _tell(self, state, transformed_fitness)

            for monitor in self.registered_hooks["post_tell"]:
                monitor.post_tell(state)

            train_info = dict(fitness=fitness, transformed_fitness=transformed_fitness)

            if has_init_ask(self.algorithm) and state.first_step:
                # this ensures that _step() will be re-jitted
                state = state.replace(generation=state.generation + 1, first_step=False)
            else:
                state = state.replace(generation=state.generation + 1)

            for monitor in self.registered_hooks["post_step"]:
                monitor.post_step(state)

            return train_info, state

        if self.jit_step:
            # the first argument is self, which should be static
            self._step = jax.jit(_step, static_argnums=(0,))
        else:
            self._step = _step

        # by default, use the first device
        self.devices = jax.local_devices()[:1]
        self.pmap_axis_name = None

    def setup(self, key):
        return State(
            StdWorkflowState(generation=0, first_step=True, rank=0, world_size=1)
        )

    def step(self, state):
        return self._step(self, state)

    def enable_multi_devices(self, state: State, pmap_axis_name=POP_AXIS_NAME) -> State:
        """
        Enable the workflow to run on multiple devices.
        Multiple nodes(processes) are also supported.
        To specify which devices are used, use env vars like `CUDA_VISIBLE_DEVICES`

        Parameters
        ----------
        state
            The state.

        Returns
        -------
        State
            The replicated state, distributed amoung all local devices
            with additional distributed information.
        """
        self.devices = jax.local_devices()
        num_devices = jax.device_count()
        num_local_devices = len(self.devices)

        self.pmap_axis_name = pmap_axis_name
        self._step = jax.pmap(
            self._step, axis_name=pmap_axis_name, static_broadcasted_argnums=0
        )

        # multi-node case
        process_id = get_process_id()
        ranks = process_id * num_local_devices + jnp.arange(
            num_local_devices, dtype=jnp.int32
        )

        state = jax.device_put_replicated(state, self.devices)
        state = state.replace(
            rank=jax.device_put_sharded(tuple(ranks), self.devices), world_size=num_devices
        )

        return state


# TODO: add mpi4jax support
# TODO: test Nvidia GPU deterministic in our parallel model, with XLA_FLAGS=--xla_gpu_deterministic_ops=true; see https://github.com/google/jax/discussions/10674
