from evox import Stateful
from evox import Algorithm, Problem
from typing import Optional, Callable


class GymPipeline(Stateful):
    def __init__(
        self,
        algorithm: Algorithm,
        problem: Problem,
        pop_transform: Optional[Callable] = None,
        fitness_transform: Optional[Callable] = None,
        monitor_update: Optional[Callable] = None
        ):
        
        self.algorithm = algorithm
        self.problem = problem
        self.pop_transform = pop_transform
        self.fitness_transform = fitness_transform
        self.monitor_update = monitor_update

    def step(self, state):
        state, init_pop = self.algorithm.ask(state)

        if self.pop_transform is not None:
            pop = self.pop_transform(init_pop)

        state, fitness = self.problem.evaluate(state, pop)
        if self.fitness_transform is not None:
            fitness = self.fitness_transform(fitness)

        if self.monitor_update is not None:
            self.monitor_update(state, self.problem, init_pop, fitness)

        state = self.algorithm.tell(state, fitness)

        return state

    def sample(self, state):
        """Sample the algorithm but don't change it's state
        """
        state_, sample_pop = self.algorithm.ask(state)
        return state, sample_pop
