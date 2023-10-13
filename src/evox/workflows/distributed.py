from typing import Callable, Optional, List, Dict, Union
from collections import deque

import chex
import jax
import jax.numpy as jnp
import numpy as np
import ray
from jax.tree_util import tree_flatten

from evox import Algorithm, Problem, State, Stateful, jit_class
from evox.utils import parse_opt_direction


@jit_class
class WorkerWorkflow(Stateful):
    def __init__(
        self,
        algorithm: Algorithm,
        problem: Problem,
        opt_direction: jax.Array,
        pop_size: int,
        start_indices: int,
        slice_sizes: int,
        pop_transform: Optional[Callable],
        fitness_transform: Optional[Callable],
    ):
        self.algorithm = algorithm
        self.problem = problem
        self.opt_direction = opt_direction
        self.pop_size = pop_size
        self.start_indices = start_indices
        self.slice_sizes = slice_sizes
        self.pop_transform = pop_transform
        self.fitness_transform = fitness_transform

    def step1(self, state: State):
        pop, state = self.algorithm.ask(state)
        assert (
            self.pop_size == pop.shape[0]
        ), f"Specified pop_size doesn't match the actual pop_size, {self.pop_size} != {pop.shape[0]}"
        partial_pop = pop[self.start_indices : self.start_indices + self.slice_sizes]

        if self.pop_transform is not None:
            partial_pop = self.pop_transform(partial_pop)

        partial_fitness, state = self.problem.evaluate(state, partial_pop)

        return partial_fitness, state

    def step2(self, state: State, fitness: List[jax.Array]):
        fitness = jnp.concatenate(fitness, axis=0)
        fitness = fitness * self.opt_direction
        if self.fitness_transform is not None:
            fitness = self.fitness_transform(fitness)

        state = self.algorithm.tell(state, fitness)
        return state

    def valid(self, state: State, metric: str):
        new_state = self.problem.valid(state, metric=metric)
        pop, new_state = self.algorithm.ask(new_state)
        partial_pop = pop[self.start_indices : self.start_indices + self.slice_sizes]

        if self.pop_transform is not None:
            partial_pop = self.pop_transform(partial_pop)

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
        opt_direction: jax.Array,
        pop_size: int,
        num_workers: int,
        options: dict,
        pop_transform: Optional[Callable],
        fitness_transform: Optional[Callable],
    ):
        assert (
            pop_size > num_workers
        ), "pop_size must be greater than the number of workers"

        # try to evenly distribute the task
        std_slice_size = pop_size // num_workers
        reminder = pop_size % num_workers
        slice_size_list = np.full(num_workers, fill_value=std_slice_size)
        slice_size_list[-reminder:] += 1

        slice_size = pop_size // num_workers
        start_index = 0
        self.workers = []
        for i, slice_size in enumerate(slice_size_list):
            self.workers.append(
                Worker.options(**options).remote(
                    WorkerWorkflow(
                        algorithm,
                        problem,
                        opt_direction,
                        pop_size,
                        start_index,
                        slice_size,
                        pop_transform,
                        fitness_transform,
                    ),
                    i,
                )
            )
            start_index += slice_size

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

    def assert_state_sync(self):
        states = ray.get([worker.get_full_state.remote() for worker in self.workers])
        leaves0, _treedef = tree_flatten(states[0])
        for state in states:
            leaves, _treedef = tree_flatten(state)
            chex.assert_trees_all_close(leaves0, leaves)
        return True


class RayDistributedWorkflow(Stateful):
    def __init__(
        self,
        algorithm: Algorithm,
        problem: Problem,
        pop_size: int,
        num_workers: int,
        monitor=None,
        opt_direction: Union[str, List[str]] = "min",
        record_pop: bool = False,
        record_time: bool = False,
        metrics: Optional[Dict[str, Callable]] = None,
        options: dict = {},
        pop_transform: Optional[Callable] = None,
        fitness_transform: Optional[Callable] = None,
        global_fitness_transform: Optional[Callable] = None,
        async_dispatch: int = 16,
    ):
        """Create a distributed workflow

        Distributed workflow can distribute the workflow to different nodes,
        it will create num_workers copies of the workflows with the same seed,
        and at each step each workflow only evaluate part of the population,
        then pass the fitness to other nodes to recreate the whole fitness array.

        pop_transform and fitness_transform are applied at each node,
        while global_fitness_transform is applied at the main node once per step,
        so monitor should be passed as global_fitness_transform.

        Parameters
        ----------
        algorithm
            The algorithm.
        problem
            The problem.
        pop_size
            Population size, this argument together with num_workers
            will be used to determine the sharding strategy.
        num_workers
            Number of workers.
        opt_direction
            The optimization direction, can be either "min" or "max"
            or a list of "min"/"max" to specific the direction for each objective.
        options
            The runtime options of the worker actor.
        pop_transform:
            Population transform, this transform is applied at each worker node.
        fitness_transform:
            Fitness transform, this transform is applied at each worker node.
        global_fitness_transform:
            This transform is applied at the main node.
        """
        opt_direction = parse_opt_direction(opt_direction)
        self.supervisor = Supervisor.remote(
            algorithm,
            problem,
            opt_direction,
            pop_size,
            num_workers,
            options,
            pop_transform,
            fitness_transform,
        )
        self.global_fitness_transform = global_fitness_transform
        self.async_dispatch_list = deque()
        self.async_dispatch = async_dispatch
        self.monitor = monitor
        self.record_pop = record_pop
        self.record_time = record_time
        self.metrics = metrics

    def setup(self, key: jax.Array):
        ray.get(self.supervisor.setup_all_workers.remote(key))
        return State()

    def step(self, state: State, block=False):
        if self.monitor and self.record_time:
            self.monitor.record_time()

        fitness, worker_futures = ray.get(self.supervisor.step.remote())

        # get the actual object
        if self.monitor is not None:
            fitness = ray.get(fitness)
            fitness = jnp.concatenate(fitness, axis=0)
            if self.metrics:
                metrics = {name: func(fitness) for name, func in self.metrics.items()}
            else:
                metrics = None
            self.monitor.record_fit(fitness, metrics)

        self.async_dispatch_list.append(worker_futures)
        while not len(self.async_dispatch_list) < self.async_dispatch:
            ray.get(self.async_dispatch_list.popleft())

        if block:
            # block until all workers have finished processing
            while len(self.async_dispatch_list) > 0:
                ray.get(self.async_dispatch_list.popleft())
        # if block is False, don't wait for step 2
        return state

    def valid(self, state: State, metric="loss"):
        fitness = ray.get(ray.get(self.supervisor.valid.remote(metric)))
        return jnp.concatenate(fitness, axis=0), state

    def sample(self, state: State):
        return ray.get(self.supervisor.sample.remote()), state

    def health_check(self, state: State):
        return ray.get(self.supervisor.assert_state_sync.remote()), state
