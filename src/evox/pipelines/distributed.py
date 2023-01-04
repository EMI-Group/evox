from typing import Callable, Optional

import chex
import jax.numpy as jnp
import numpy as np
import ray
from evox import Algorithm, Problem, State, Stateful
from evox.monitors import FitnessMonitor, PopulationMonitor
from jax.tree_util import tree_flatten


class WorkerPipeline(Stateful):
    def __init__(
        self,
        algorithm: Algorithm,
        problem: Problem,
        pop_size: int,
        start_indices: int,
        slice_sizes: int,
        pop_transform: Optional[Callable],
        fitness_transform: Optional[Callable],
    ):
        self.algorithm = algorithm
        self.problem = problem
        self.pop_size = pop_size
        self.start_indices = start_indices
        self.slice_sizes = slice_sizes
        self.pop_transform = pop_transform
        self.fitness_transform = fitness_transform

    def step1(self, state):
        state, pop = self.algorithm.ask(state)
        assert (
            self.pop_size == pop.shape[0]
        ), f"Specified pop_size doesn't match the actual pop_size, {self.pop_size} != {pop.shape[0]}"
        partial_pop = pop[self.start_indices : self.start_indices + self.slice_sizes]

        if self.pop_transform is not None:
            partial_pop = self.pop_transform(partial_pop)

        state, partial_fitness = self.problem.evaluate(state, partial_pop)

        return state, partial_fitness

    def step2(self, state, fitness):
        if self.fitness_transform is not None:
            fitness = self.fitness_transform(fitness)

        state = self.algorithm.tell(state, fitness)
        return state

    def valid(self, state, metric):
        new_state = self.problem.valid(state, metric=metric)
        new_state, pop = self.algorithm.ask(new_state)
        partial_pop = pop[self.start_indices : self.start_indices + self.slice_sizes]

        if self.pop_transform is not None:
            partial_pop = self.pop_transform(partial_pop)

        new_state, partial_fitness = self.problem.evaluate(new_state, partial_pop)
        return state, partial_fitness

    def sample(self, state):
        return self.algorithm.ask(state)


@ray.remote
class Worker:
    def __init__(self, pipeline, worker_index):
        self.pipeline = pipeline
        self.worker_index = worker_index

    def init(self, key):
        self.state = self.pipeline.init(key)
        self.initialized = True

    def step1(self):
        self.state, parital_fitness = self.pipeline.step1(self.state)
        return parital_fitness

    def step2(self, fitness):
        fitness = ray.get(fitness)
        fitness = jnp.concatenate(fitness, axis=0)
        self.state = self.pipeline.step2(self.state, fitness)

    def valid(self, metric):
        _state, fitness = self.pipeline.valid(self.state, metric)
        return fitness

    def get_full_state(self):
        return self.state

    def sample(self):
        _state, sample = self.pipeline.sample(self.state)
        return sample


@ray.remote
class Supervisor:
    def __init__(
        self,
        algorithm: Algorithm,
        problem: Problem,
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
                    WorkerPipeline(
                        algorithm,
                        problem,
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

    def setup_all_workers(self, key):
        ray.get([worker.init.remote(key) for worker in self.workers])

    def sample(self):
        return ray.get(self.workers[0].sample.remote())

    def step(self):
        fitness = [worker.step1.remote() for worker in self.workers]
        worker_futures = [worker.step2.remote(fitness) for worker in self.workers]
        return fitness, worker_futures

    def valid(self, metric):
        fitness = [worker.valid.remote(metric) for worker in self.workers]
        return fitness

    def assert_state_sync(self):
        states = ray.get([worker.get_full_state.remote() for worker in self.workers])
        leaves0, _treedef = tree_flatten(states[0])
        for state in states:
            leaves, _treedef = tree_flatten(state)
            chex.assert_trees_all_close(leaves0, leaves)
        return True


class DistributedPipeline(Stateful):
    def __init__(
        self,
        algorithm: Algorithm,
        problem: Problem,
        pop_size: int,
        num_workers: int,
        options: dict = {},
        pop_transform: Optional[Callable] = None,
        fitness_transform: Optional[Callable] = None,
        global_fitness_transform: Optional[Callable] = None,
    ):
        """Create a distributed pipeline

        Distributed pipeline can distribute the pipeline to different nodes,
        it will create num_workers copies of the pipelines with the same seed,
        and at each step each pipeline only evaluate part of the population,
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
        options
            The runtime options of the worker actor.
        pop_transform:
            Population transform, this transform is applied at each worker node.
        fitness_transform:
            Fitness transform, this transform is applied at each worker node.
        global_fitness_transform:
            This transform is applied at the main node.
        """
        self.supervisor = Supervisor.remote(
            algorithm,
            problem,
            pop_size,
            num_workers,
            options,
            pop_transform,
            fitness_transform,
        )
        self.global_fitness_transform = global_fitness_transform

    def setup(self, key):
        ray.get(self.supervisor.setup_all_workers.remote(key))
        return State()

    def step(self, state):
        fitness, worker_futures = ray.get(self.supervisor.step.remote())
        # get the actual object
        fitness = ray.get(fitness)
        fitness = jnp.concatenate(fitness, axis=0)
        if self.global_fitness_transform is not None:
            fitness = self.global_fitness_transform(fitness)
        # block until all workers have finished processing
        ray.get(worker_futures)
        return state

    def valid(self, state, metric="loss"):
        fitness = ray.get(ray.get(self.supervisor.valid.remote(metric)))
        return state, jnp.concatenate(fitness, axis=0)

    def sample(self, state):
        return state, ray.get(self.supervisor.sample.remote())

    def health_check(self, state):
        return state, ray.get(self.supervisor.assert_state_sync.remote())
