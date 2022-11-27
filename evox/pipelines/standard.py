from evox import Stateful
from evox import Algorithm, Problem
from evox.monitors import FitnessMonitor, PopulationMonitor
from typing import Optional, Callable


class StdPipeline(Stateful):
    def __init__(
        self,
        algorithm: Algorithm,
        problem: Problem,
        pop_transform: Optional[Callable] = None,
        fitness_transform: Optional[Callable] = None):
        self.algorithm = algorithm
        self.problem = problem
        self.pop_transform = pop_transform
        self.fitness_transform = fitness_transform

    def step(self, state):
        state, pop = self.algorithm.ask(state)
        if self.pop_transform is not None:
            pop = self.pop_transform(pop)

        state, fitness = self.problem.evaluate(state, pop)
        if self.fitness_transform is not None:
            fitness = self.fitness_transform(fitness)

        state = self.algorithm.tell(state, fitness)

        return state

    def valid(self, state, metric="loss"):
        new_state = self.problem.valid(state, metric=metric)
        new_state, pop = self.algorithm.ask(new_state)
        if self.pop_transform is not None:
            pop = self.pop_transform(pop)

        new_state, fitness = self.problem.evaluate(new_state, pop)
        return state, fitness

    def sample(self, state):
        """Sample the algorithm but don't change it's state
        """
        state_, sample_pop = self.algorithm.ask(state)
        return state, sample_pop
