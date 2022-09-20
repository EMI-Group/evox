from evoxlib import Module
from evoxlib import Algorithm, Problem
from evoxlib.monitors import FitnessMonitor, PopulationMonitor


class StdPipeline(Module):
    def __init__(self, algorithm: Algorithm, problem: Problem, fitness_monitor: bool = False, population_monitor: bool = False):
        self.algorithm = algorithm
        self.problem = problem

        if fitness_monitor:
            self.fitness_monitor = FitnessMonitor()
        else:
            self.fitness_monitor = None

        if population_monitor:
            self.population_monitor = PopulationMonitor(2)
        else:
            self.population_monitor = None

    def step(self, state):
        state, pop = self.algorithm.ask(state)
        state, fitness = self.problem.evaluate(state, pop)
        state = self.algorithm.tell(state, fitness)

        if self.fitness_monitor is not None:
            self.fitness_monitor.update(fitness)
        if self.population_monitor is not None:
            self.population_monitor.update(pop)

        return state

    def get_min_fitness(self, state):
        if self.fitness_monitor is None:
            raise ValueError("Fitness monitor is required")

        return state, self.fitness_monitor.get_min_fitness()
