from evox import jit_method, Stateful, Algorithm, Problem
from typing import Optional, Callable, Union
import jax
import jax.numpy as jnp
from jax import jit
from jax.tree_util import tree_map
from jax.sharding import PositionalSharding
from jaxlib.xla_extension import PjitFunction
import jax.experimental.host_callback as hcb


class UniWorkflow(Stateful):
    """Experimental unified workflow, designed to provide unparallel performance for EC workflow.

    Provide automatic multi-device (e.g. multiple gpus) computation as well as distributed computation
    using JAX's native components.

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
        jitted_problem: Union[bool, str] = True,
        num_objectives=None,
    ):
        self.algorithm = algorithm
        self.problem = problem
        self.monitor = monitor
        self.pop_transform = pop_transform
        self.fit_transform = fit_transform
        self.jitted_problem = jitted_problem
        self.num_objectives = num_objectives
        self.distributed_step = False
        assert not (
            jitted_problem == False and self.num_objectives == False
        ), "When using external problem, num_objectives must be set"

        def _step(self, state):
            pop, state = self.algorithm.ask(state)

            if self.monitor is not None:
                hcb.id_tap(self.monitor.record_pop, pop)
            if self.distributed_step is True:
                pop = jax.lax.dynamic_slice_in_dim(
                    pop, self.start_index, self.slice_size, axis=0
                )
            if self.pop_transform is not None:
                pop = self.pop_transform(pop)

            # if the function is jitted
            if self.jitted_problem:
                fitness, state = self.problem.evaluate(state, pop)
            else:
                pop_size = pop.shape[0]
                fitness, state = hcb.call(
                    lambda state_pop: self.problem.evaluate(state_pop[0], state_pop[1]),
                    (state, pop),
                    result_shape=(
                        jax.ShapeDtypeStruct(
                            (pop_size, self.num_objectives), dtype=jnp.float32
                        ),
                        state,
                    ),
                )
            if self.distributed_step is True:
                fitness = jax.lax.all_gather(fitness, "i", axis=0, tiled=True)

            if self.monitor is not None:
                hcb.id_tap(self.monitor.record_fit, fitness)

            if self.fit_transform is not None:
                fitness = self.fit_transform(fitness)

            state = self.algorithm.tell(state, fitness)

            return state

        self._step = jit_method(_step)

    def step(self, state):
        return self._step(self, state)

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

    def enable_multi_devices(self, state):
        dummy_pop, _ = jax.eval_shape(self.algorithm.ask, state)
        pop_size, dim = dummy_pop.shape
        sharding = PositionalSharding(jax.local_devices()).reshape(
            1, jax.local_device_count()
        )
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
        dummy_pop, _ = jax.eval_shape(self.algorithm.ask, state)
        pop_size, dim = dummy_pop.shape
        self.slice_size = pop_size // jax.process_count()
        self.start_index = self.slice_size * jax.process_index()
        self.distributed_step = True
        return state
