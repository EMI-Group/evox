import warnings
from collections import deque
from typing import Callable, Dict, List, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
import ray
from jax import jit
from jax.tree_util import tree_flatten

from evox import Algorithm, Monitor, Problem, State, Stateful, jit_class
from evox.utils import algorithm_has_init_ask, parse_opt_direction


class WorkerWorkflow(Stateful):
    def __init__(
        self,
        algorithm: Algorithm,
        problem: Problem,
        num_workers: int,
        monitor_actor,
        non_empty_hooks: List[str],
        opt_direction: jax.Array,
        worker_index: int,
        sol_transforms: List[Callable],
        fit_transforms: List[Callable],
    ):
        self.algorithm = algorithm
        self.problem = problem
        self.num_workers = num_workers
        self.monitor_actor = monitor_actor
        self.non_empty_hooks = non_empty_hooks
        self.opt_direction = opt_direction
        self.worker_index = worker_index
        self.sol_transforms = sol_transforms
        self.fit_transforms = fit_transforms

    def setup(self, key):
        return State(generation=0)

    def _get_slice(self, pop_size):
        slice_per_worker = pop_size // self.num_workers
        remainder = pop_size % self.num_workers
        start = slice_per_worker * self.worker_index + min(self.worker_index, remainder)
        end = start + slice_per_worker + (self.worker_index < remainder)
        return start, end

    def step1(self, state: State):
        if "pre_ask" in self.non_empty_hooks:
            ray.get(self.monitor_actor.push.remote("pre_ask", state))

        if state.generation == 0:
            is_init = algorithm_has_init_ask(self.algorithm, state)
        else:
            is_init = False

        if is_init:
            cand_sol, state = self.algorithm.init_ask(state)
        else:
            cand_sol, state = self.algorithm.ask(state)

        if "post_ask" in self.non_empty_hooks:
            ray.get(self.monitor_actor.push.remote("post_ask", None, cand_sol))

        start, end = self._get_slice(cand_sol.shape[0])
        partial_sol = cand_sol[start:end]

        transformed_partial_sol = partial_sol
        for transform in self.sol_transforms:
            transformed_partial_sol = transform(transformed_partial_sol)

        if "pre_eval" in self.non_empty_hooks:
            ray.get(
                self.monitor_actor.push.remote(
                    "pre_eval", state, partial_sol, transformed_partial_sol
                )
            )

        partial_fitness, state = self.problem.evaluate(state, transformed_partial_sol)

        return partial_fitness, state

    def step2(self, state: State, fitness: List[jax.Array]):
        if state.generation == 0:
            is_init = algorithm_has_init_ask(self.algorithm, state)
        else:
            is_init = False

        fitness = jnp.concatenate(fitness, axis=0)
        fitness = fitness * self.opt_direction

        if "post_eval" in self.non_empty_hooks:
            ray.get(
                self.monitor_actor.push.remote("post_eval", None, None, None, fitness)
            )

        transformed_fitness = fitness
        for transform in self.fit_transforms:
            transformed_fitness = transform(transformed_fitness)

        if "pre_tell" in self.non_empty_hooks:
            ray.get(
                self.monitor_actor.push.remote(
                    "pre_tell",
                    state,
                    None,
                    None,
                    fitness,
                    transformed_fitness,
                )
            )

        if is_init:
            state = self.algorithm.init_tell(state, fitness)
        else:
            state = self.algorithm.tell(state, fitness)

        if "post_tell" in self.non_empty_hooks:
            ray.get(self.monitor_actor.push.remote("post_tell", state))

        return state.update(generation=state.generation + 1)

    def valid(self, state: State, metric: str):
        new_state = self.problem.valid(state, metric=metric)
        pop, new_state = self.algorithm.ask(new_state)
        partial_pop = pop[self.start_indices : self.start_indices + self.slice_sizes]

        partial_fitness, new_state = self.problem.evaluate(new_state, partial_pop)
        return partial_fitness, state

    def sample(self, state: State):
        return self.algorithm.ask(state)


@ray.remote
class Worker:
    def __init__(self, workflow: WorkerWorkflow, worker_index: int):
        self.workflow = workflow
        self.worker_index = worker_index

    def init(self, key: jax.Array):
        self.state = self.workflow.init(key)
        self.initialized = True

    def step1(self):
        parital_fitness, self.state = self.workflow.step1(self.state)
        return parital_fitness

    def step2(self, fitness: jax.Array):
        fitness = ray.get(fitness)
        self.state = self.workflow.step2(self.state, fitness)

    def valid(self, metric: str):
        fitness, _state = self.workflow.valid(self.state, metric)
        return fitness

    def get_full_state(self):
        return self.state

    def sample(self):
        sample, _state = self.workflow.sample(self.state)
        return sample


@ray.remote
class Supervisor:
    def __init__(
        self,
        algorithm: Algorithm,
        problem: Problem,
        num_workers: int,
        monitor_actor,
        non_empty_hooks: List[str],
        opt_direction: jax.Array,
        options: dict,
        sol_transforms: List[Callable],
        fit_transforms: List[Callable],
    ):
        # try to evenly distribute the task
        self.workers = []
        for i in range(num_workers):
            self.workers.append(
                Worker.options(**options).remote(
                    WorkerWorkflow(
                        algorithm,
                        problem,
                        num_workers,
                        # only the first worker has monitors
                        monitor_actor if i == 0 else None,
                        non_empty_hooks if i == 0 else [],
                        opt_direction,
                        i,
                        sol_transforms,
                        fit_transforms,
                    ),
                    i,
                )
            )

    def setup_all_workers(self, key: jax.Array):
        ray.get([worker.init.remote(key) for worker in self.workers])

    def sample(self):
        return ray.get(self.workers[0].sample.remote())

    def step(self):
        fitness = [worker.step1.remote() for worker in self.workers]
        worker_futures = [worker.step2.remote(fitness) for worker in self.workers]
        return fitness, worker_futures

    def valid(self, metric: str):
        fitness = [worker.valid.remote(metric) for worker in self.workers]
        return fitness


@ray.remote
class MonitorActor:
    def __init__(self):
        self.call_queue = []

    def push(self, hook, *args, **kwargs):
        self.call_queue.append((hook, args, kwargs))

    def get_call_queue(self):
        call_queue = self.call_queue
        self.call_queue = []
        return call_queue


class RayDistributedWorkflow(Stateful):
    def __init__(
        self,
        algorithm: Algorithm,
        problem: Problem,
        num_workers: int,
        monitors=[],
        opt_direction: Union[str, List[str]] = "min",
        metrics: Optional[Dict[str, Callable]] = None,
        options: dict = {},
        sol_transforms: List[Callable] = [],
        fit_transforms: List[Callable] = [],
        global_fit_transform: List[Callable] = [],
        async_dispatch: int = 4,
        monitor=None,
    ):
        """Create a distributed workflow

        Distributed workflow can distribute the workflow to different nodes,
        it will create num_workers copies of the workflows with the same seed,
        and at each step each workflow only evaluate part of the population,
        then pass the fitness to other nodes to recreate the whole fitness array.

        sol_transforms and fit_transforms are applied at each node,

        Parameters
        ----------
        algorithm
            The algorithm.
        problem
            The problem.
        num_workers
            Number of workers.
        opt_direction
            The optimization direction, can be either "min" or "max"
            or a list of "min"/"max" to specific the direction for each objective.
        options
            The runtime options of the worker actor.
        sol_transforms:
            Population transform, this transform is applied at each worker node.
        fit_transforms:
            Fitness transform, this transform is applied at each worker node.
        """
        self.async_dispatch_list = deque()
        self.async_dispatch = async_dispatch
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
        self.monitors = monitors
        if monitor is not None:
            warnings.warn(
                "`monitor` is deprecated, use the `monitors` parameter with a list of monitors instead",
                DeprecationWarning,
            )
            self.monitors = [monitor]
        for monitor in self.monitors:
            hooks = monitor.hooks()
            for hook in hooks:
                self.registered_hooks[hook].append(monitor)

        self.opt_direction = parse_opt_direction(opt_direction)
        for monitor in self.monitors:
            monitor.set_opt_direction(self.opt_direction)

        non_empty_hooks = [
            hook
            for hook in self.registered_hooks
            if len(self.registered_hooks[hook]) > 0
        ]

        self.monitor_actor = MonitorActor.remote()
        self.supervisor = Supervisor.remote(
            algorithm,
            problem,
            num_workers,
            self.monitor_actor,
            non_empty_hooks,
            self.opt_direction,
            options,
            sol_transforms,
            fit_transforms,
        )

    def setup(self, key: jax.Array):
        ray.get(self.supervisor.setup_all_workers.remote(key))
        return State()

    def step(self, state: State, block=False):
        for monitor in self.registered_hooks["pre_step"]:
            monitor.pre_step(state)

        fitness, worker_futures = ray.get(self.supervisor.step.remote())

        # get the actual object
        fitness = ray.get(fitness)
        fitness = jnp.concatenate(fitness, axis=0)

        for monitor in self.registered_hooks["post_eval"]:
            monitor.post_eval(state, None, None, fitness)

        self.async_dispatch_list.append(worker_futures)
        while not len(self.async_dispatch_list) < self.async_dispatch:
            ray.get(self.async_dispatch_list.popleft())

        if block:
            # block until all workers have finished processing
            while len(self.async_dispatch_list) > 0:
                ray.get(self.async_dispatch_list.popleft())
        # if block is False, don't wait for step 2

        monitor_calls = ray.get(self.monitor_actor.get_call_queue.remote())
        for hook, args, kwargs in monitor_calls:
            for monitor in self.registered_hooks[hook]:
                getattr(monitor, hook)(*args, **kwargs)

        for monitor in self.registered_hooks["post_step"]:
            monitor.post_step(state)

        return state

    def valid(self, state: State, metric="loss"):
        fitness = ray.get(ray.get(self.supervisor.valid.remote(metric)))
        return jnp.concatenate(fitness, axis=0), state

    def sample(self, state: State):
        return ray.get(self.supervisor.sample.remote()), state
