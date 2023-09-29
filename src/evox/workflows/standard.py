from typing import Callable, Dict, Optional

from evox import Algorithm, Problem, Stateful


class StdWorkflow(Stateful):
    def __init__(
        self,
        algorithm: Algorithm,
        problem: Problem,
        monitor=None,
        pop_transform: Optional[Callable] = None,
        fitness_transform: Optional[Callable] = None,
        record_pop: bool = False,
        record_time: bool = False,
        metrics: Optional[Dict[str, Callable]] = None,
    ):
        self.algorithm = algorithm
        self.problem = problem
        self.monitor = monitor
        self.record_pop = record_pop
        self.record_time = record_time
        self.metrics = metrics
        self.pop_transform = pop_transform
        self.fitness_transform = fitness_transform

    def step(self, state):
        if self.monitor and self.record_time:
            self.monitor.record_time()

        pop, state = self.algorithm.ask(state)

        if self.monitor and self.record_pop:
            self.monitor.record_pop(pop)

        if self.pop_transform is not None:
            pop = self.pop_transform(pop)

        fitness, state = self.problem.evaluate(state, pop)

        if self.monitor:
            if self.metrics:
                metrics = {name: func(fitness) for name, func in self.metrics.items()}
            else:
                metrics = None
            self.monitor.record_fit(fitness, metrics)

        if self.fitness_transform is not None:
            fitness = self.fitness_transform(fitness)

        state = self.algorithm.tell(state, fitness)

        return state

    def valid(self, state, metric="loss"):
        new_state = self.problem.valid(state, metric=metric)
        pop, new_state = self.algorithm.ask(new_state)
        if self.pop_transform is not None:
            pop = self.pop_transform(pop)

        new_state, fitness = self.problem.evaluate(new_state, pop)
        return fitness, state

    def sample(self, state):
        """Sample the algorithm but don't change it's state"""
        sample_pop, state_ = self.algorithm.ask(state)
        if self.pop_transform is not None:
            sample_pop = self.pop_transform(sample_pop)

        return sample_pop, state
