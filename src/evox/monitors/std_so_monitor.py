import jax.numpy as jnp
import warnings
import jax.experimental.host_callback as hcb


class StdSOMonitor:
    """Standard single-objective monitor
    Used for single-objective workflow,
    can monitor fitness and the population.

    Parameters
    ----------
    record_topk
        Control how many elite solutions are recorded.
        Default is 1, which will record the best individual.
    record_fit_history
        Whether to record the full history of fitness value.
        Default to True. Setting it to False may reduce memory usage.
    """

    def __init__(self, record_topk=1, record_fit_history=True):
        self.record_fit_history = record_fit_history
        self.fitness_history = []
        self.record_topk = record_topk
        self.current_population = None
        self.topk_solutions = None
        self.topk_fitness = None
        self.opt_direction = 1  # default to min, so no transformation is needed

    def set_opt_direction(self, opt_direction):
        self.opt_direction = opt_direction

    def record_pop(self, pop, tranform=None):
        self.current_population = pop

    def record_fit(self, fitness, metrics=None, transform=None):
        if self.record_fit_history:
            self.fitness_history.append(fitness)
        if self.record_topk == 1:
            # handle the case where topk = 1
            # don't need argsort / top_k, which are slower
            current_min_fit = jnp.min(fitness, keepdims=True)
            if self.topk_fitness is None or self.topk_fitness > current_min_fit:
                self.topk_fitness = current_min_fit
                if self.current_population is not None:
                    individual_index = jnp.argmin(fitness)
                    # use slice to keepdim,
                    # because topk_solutions should have dim of (1, dim)
                    self.topk_solutions = self.current_population[
                        individual_index : individual_index + 1
                    ]
        else:
            # since topk > 1, we have to sort the fitness
            if self.topk_fitness is None:
                self.topk_fitness = fitness
            else:
                self.topk_fitness = jnp.concatenate([self.topk_fitness, fitness])

            if self.current_population is not None:
                if self.topk_solutions is None:
                    self.topk_solutions = self.current_population
                else:
                    self.topk_solutions = jnp.concatenate(
                        [self.topk_solutions, self.current_population], axis=0
                    )
            rank = jnp.argsort(self.topk_fitness)
            topk_rank = rank[: self.record_topk]
            if self.current_population is not None:
                self.topk_solutions = self.topk_solutions[topk_rank]
            self.topk_fitness = self.topk_fitness[topk_rank]

    def get_last(self):
        return self.opt_direction * self.fitness_history[-1]

    def get_topk_fitness(self):
        return self.opt_direction * self.topk_fitness

    def get_topk_solutions(self):
        return self.topk_solutions

    def get_best_fitness(self):
        if self.topk_fitness is None:
            warnings.warn("trying to get info from a monitor with no recorded data")
            return None
        return self.opt_direction * self.topk_fitness[0]

    def get_best_solution(self):
        if self.topk_solutions is None:
            warnings.warn("trying to get info from a monitor with no recorded data")
            return None
        return self.topk_solutions[0]

    def get_history(self):
        return [self.opt_direction * fit for fit in self.fitness_history]

    def flush(self):
        hcb.barrier_wait()

    def close(self):
        self.flush()
