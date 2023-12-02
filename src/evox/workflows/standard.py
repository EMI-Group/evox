from typing import Callable, Dict, Optional, Union, List

from evox import Algorithm, Problem, Stateful, State
from evox.utils import parse_opt_direction, algorithm_has_init_ask


class StdWorkflow(Stateful):
    def __init__(
        self,
        algorithm: Algorithm,
        problem: Problem,
        monitor=None,
        opt_direction: Union[str, List[str]] = "min",
        pop_transform: Optional[Callable] = None,
        fitness_transform: Optional[Callable] = None,
        record_pop: bool = False,
        record_time: bool = False,
        metrics: Optional[Dict[str, Callable]] = None,
    ):
        """
        Parameters
        ----------
        algorithm
            The algorithm.
        problem
            The problem.
        monitor
            Optional monitor.
        opt_direction
            The optimization direction, can be either "min" or "max"
            or a list of "min"/"max" to specific the direction for each objective.
        pop_transform
            Optional population transform function,
            usually used to decode the population
            into the format that can be understood by the problem.
        fit_transform
            Optional fitness transform function.
            usually used to apply fitness shaping.
        record_pop
            Whether to record the population if monitor is enabled.
        record_time
            Whether to record the time at the end of each generation.
            Due to its timing nature,
            record_time requires synchronized functional call.
            Default to False.
        """
        self.algorithm = algorithm
        self.problem = problem
        self.monitor = monitor
        self.record_pop = record_pop
        self.record_time = record_time
        self.metrics = metrics
        self.pop_transform = pop_transform
        self.fitness_transform = fitness_transform
        self.opt_direction = parse_opt_direction(opt_direction)
        self.monitor.set_opt_direction(self.opt_direction)

    def setup(self, key):
        return State(generation=0)

    def step(self, state):
        is_init = False
        if state.generation == 0:
            is_init = algorithm_has_init_ask(self.algorithm, state)
        else:
            is_init = False

        if self.monitor and self.record_time:
            self.monitor.record_time()

        if is_init:
            pop, state = self.algorithm.init_ask(state)
        else:
            pop, state = self.algorithm.ask(state)

        if self.monitor and self.record_pop:
            self.monitor.record_pop(pop)

        if self.pop_transform is not None:
            pop = self.pop_transform(pop)

        fitness, state = self.problem.evaluate(state, pop)

        fitness = fitness * self.opt_direction

        if self.monitor:
            if self.metrics:
                metrics = {name: func(fitness) for name, func in self.metrics.items()}
            else:
                metrics = None
            self.monitor.record_fit(fitness, metrics)

        if self.fitness_transform is not None:
            fitness = self.fitness_transform(fitness)

        if is_init:
            state = self.algorithm.init_tell(state, fitness)
        else:
            state = self.algorithm.tell(state, fitness)

        return state.update(generation=state.generation + 1)

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
