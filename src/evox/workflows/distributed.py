import warnings
from collections import deque
from typing import Callable, Dict, List, Optional, Union

import jax
import jax.numpy as jnp
import ray

from evox import (
    Algorithm,
    Problem,
    State,
    Workflow,
    use_state,
    has_init_ask,
    has_init_tell,
)
from evox.utils import parse_opt_direction


class WorkerWorkflow(Workflow):
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
        return State(generation=0, first_step=True)

    def _get_slice(self, pop_size):
        slice_per_worker = pop_size // self.num_workers
        remainder = pop_size % self.num_workers
        start = slice_per_worker * self.worker_index + min(self.worker_index, remainder)
        end = start + slice_per_worker + (self.worker_index < remainder)
        return start, end

    def _ask(self, state):
        if has_init_ask(self.algorithm) and state.first_step:
            ask = self.algorithm.init_ask
        else:
            ask = self.algorithm.ask

        # candidate: individuals that need to be evaluated (may differ from population)
        # Note: num_cands can be different from init_ask() and ask()
        cands, state = use_state(ask)(state)

        return cands, state

    def step1(self, state: State):
        if "pre_ask" in self.non_empty_hooks:
            ray.get(self.monitor_actor.push.remote("pre_ask", state))

        cand_sol, state = self._ask(state)
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

        partial_fitness, state = use_state(self.problem.evaluate)(
            state, transformed_partial_sol
        )

        return partial_fitness, state

    def _tell(self, state, transformed_fitness):
        if has_init_tell(self.algorithm) and state.first_step:
            tell = self.algorithm.init_tell
        else:
            tell = self.algorithm.tell

        state = use_state(tell)(state, transformed_fitness)

        return state

    def step2(self, state: State, fitness: List[jax.Array]):
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

        state = self._tell(state, fitness)

        if "post_tell" in self.non_empty_hooks:
            ray.get(self.monitor_actor.push.remote("post_tell", state))

        if has_init_ask(self.algorithm) and state.first_step:
            # this ensures that _step() will be re-jitted
            state = state.replace(generation=state.generation + 1, first_step=False)
        else:
            state = state.replace(generation=state.generation + 1)

        return state

    def valid(self, state: State, metric: str):
        new_state = use_state(self.problem.valid)(state, metric=metric)
        pop, new_state = use_state(self.algorithm.ask)(new_state)
        partial_pop = pop[self.start_indices : self.start_indices + self.slice_sizes]

        partial_fitness, new_state = use_state(self.problem.evaluate)(
            new_state, partial_pop
        )
        return partial_fitness, state

    def sample(self, state: State):
        return use_state(self.algorithm.ask)(state)


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


class RayDistributedWorkflow(Workflow):
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
        auto_exec_callbacks=True,
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
        for i, monitor in enumerate(self.monitors):
            hooks = monitor.hooks()
            for hook in hooks:
                self.registered_hooks[hook].append(monitor)
            setattr(self, f"monitor{i}", monitor)

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
        self.auto_exec_callbacks = auto_exec_callbacks

    def setup(self, key: jax.Array):
        ray.get(self.supervisor.setup_all_workers.remote(key))
        return State()

    def step(self, state: State, block=False):
        state = self._pre_step_hook(state)

        fitness, worker_futures = ray.get(self.supervisor.step.remote())

        # get the actual object
        fitness = ray.get(fitness)
        fitness = jnp.concatenate(fitness, axis=0)

        state = self._post_eval_hook(state, fitness)

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

        state = self._post_step_hook(state)

        if self.auto_exec_callbacks:
            state = state.execute_callbacks(state)

        return state

    def valid(self, state: State, metric="loss"):
        fitness = ray.get(ray.get(self.supervisor.valid.remote(metric)))
        return jnp.concatenate(fitness, axis=0), state

    def sample(self, state: State):
        return ray.get(self.supervisor.sample.remote()), state

    def _pre_step_hook(self, state):
        for monitor in self.registered_hooks["pre_step"]:
            state = use_state(monitor.pre_step)(state, state)
        return state

    def _pre_ask_hook(self, state):
        for monitor in self.registered_hooks["pre_ask"]:
            state = use_state(monitor.pre_ask)(state, state)
        return state

    def _post_ask_hook(self, state, cands):
        for monitor in self.registered_hooks["post_ask"]:
            state = use_state(monitor.post_ask)(state, state, cands)
        return state

    def _pre_eval_hook(self, state, transformed_cands):
        for monitor in self.registered_hooks["pre_eval"]:
            state = use_state(monitor.pre_eval)(state, state, transformed_cands)
        return state

    def _post_eval_hook(self, state, fitness):
        for monitor in self.registered_hooks["post_eval"]:
            state = use_state(monitor.post_eval)(state, state, fitness)
        return state

    def _pre_tell_hook(self, state, transformed_fitness):
        for monitor in self.registered_hooks["pre_tell"]:
            state = use_state(monitor.pre_tell)(state, state, transformed_fitness)
        return state

    def _post_tell_hook(self, state):
        for monitor in self.registered_hooks["post_tell"]:
            state = use_state(monitor.post_tell)(state, state)
        return state

    def _post_step_hook(self, state):
        for monitor in self.registered_hooks["post_step"]:
            state = use_state(monitor.post_step)(state, state)
        return state
