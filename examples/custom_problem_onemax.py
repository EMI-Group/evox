# An exmaple of implement custom problem in EvoX.
# Here, we implement the OneMax problem, the fitness is defined as the sum of all the digits in a bitstring.
# For example, "100111" -> 4, "000101" -> 2.
# The goal is to find the bitstring that maximize the fitness.
# Since in EvoX, algorithms try to minimize the fitness, so we return the negitive sum as our fitness.

import jax.numpy as jnp
from evox import Problem, jit_class


@jit_class
class OneMax(Problem):
    def __init__(self, neg_fitness=True) -> None:
        super().__init__()
        self.neg_fitess = neg_fitness

    def evaluate(self, state, bitstrings):
        # bitstrings has shape (pop_size, num_bits)
        # so sum along the axis 1.
        fitness = jnp.sum(bitstrings, axis=1)
        # Since in EvoX, algorithms try to minimize the fitness
        # so return the negitive value.
        if self.neg_fitess:
            fitness = -fitness
        return fitness, state
