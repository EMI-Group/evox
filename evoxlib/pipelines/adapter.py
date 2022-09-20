from evoxlib import Module
from evoxlib import Algorithm, Problem
from evoxlib.monitors import FitnessMonitor


class AdapterPipeline(Module):
    def __init__(self, algorithm: Algorithm, problem: Problem, adapter, fitness_monitor: bool = False):
        self.algorithm = algorithm
        self.problem = problem
        self.adapter = adapter

        if fitness_monitor:
            self.fitness_monitor = FitnessMonitor()
        else:
            self.fitness_monitor = None

    def step(self, state):
        state, pop = self.algorithm.ask(state)
        adapted_pop = self.adapter.to_tree(pop, True)
        state, fitness = self.problem.evaluate(state, adapted_pop)
        state = self.algorithm.tell(state, fitness)

        if self.fitness_monitor is not None:
            self.fitness_monitor.update(fitness)

        return state

    def get_min_fitness(self, state):
        if self.fitness_monitor is None:
            raise ValueError("Fitness monitor is required")

        return state, self.fitness_monitor.get_min_fitness()
