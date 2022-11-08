from evox import Stateful, State
from evox import Algorithm, Problem
from evox.monitors import FitnessMonitor, PopulationMonitor
import ray
import jax.numpy as jnp
from jax.tree_util import tree_flatten
import chex
from typing import Optional, Callable


class WorkerPipeline(Stateful):
    def __init__(
        self,
        algorithm: Algorithm,
        problem: Problem,
        start_indices: int,
        slice_sizes: int,
        pop_transform: Optional[Callable],
        fitness_transform: Optional[Callable],
    ):
        self.algorithm = algorithm
        self.problem = problem
        self.start_indices = start_indices
        self.slice_sizes = slice_sizes
        self.pop_transform = pop_transform
        self.fitness_transform = fitness_transform

    def step1(self, state):
        state, pop = self.algorithm.ask(state)
        if self.fitness_transform is not None:
            pop = self.fitness_transform(pop)

        partial_pop = pop[self.start_indices : self.start_indices + self.slice_sizes]

        if self.pop_transform is not None:
            partial_pop = self.pop_transform(partial_pop)

        state, partial_fitness = self.problem.evaluate(state, partial_pop)

        return state, partial_fitness

    def step2(self, state, fitness):
        state = self.algorithm.tell(state, fitness)
        return state

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
        assert pop_size % num_workers == 0
        slice_size = pop_size // num_workers
        self.workers = [
            Worker.options(**options).remote(
                WorkerPipeline(
                    algorithm,
                    problem,
                    i * slice_size,
                    slice_size,
                    pop_transform,
                    fitness_transform,
                ),
                i,
            )
            for i in range(num_workers)
        ]

    def setup_all_workers(self, key):
        ray.get([worker.init.remote(key) for worker in self.workers])

    def sample(self):
        return ray.get(self.workers[0].sample.remote())

    def step(self):
        fitness = [worker.step1.remote() for worker in self.workers]
        worker_futures = [worker.step2.remote(fitness) for worker in self.workers]
        return fitness, worker_futures

    def assert_state_sync(self):
        states = ray.get([worker.get_full_state.remote() for worker in self.workers])
        leaves0, _treedef = tree_flatten(states[0])
        for state in states:
            leaves, _treedef = tree_flatten(state)
            chex.assert_trees_all_close(leaves0, leaves)


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

    def sample(self, state):
        return state, ray.get(self.supervisor.sample.remote())

    def health_check(self, state):
        ray.get(self.supervisor.assert_state_sync.remote())
