# An example of implementing genetic algorithm that solves OneMax problem in EvoX.
# This algorithm uses binary crossover and bitflip mutation.

from evox import Algorithm, State, jit_class
from evox.operators import mutation, crossover, selection
from jax import random
import jax.numpy as jnp


@jit_class
class ExampleGA(Algorithm):
    def __init__(self, pop_size, ndim, flip_prob):
        super().__init__()
        # those are hyperparameters that stay fixed.
        self.pop_size = pop_size
        self.ndim = ndim
        # the probability of fliping each bit
        self.flip_prob = flip_prob

    def setup(self, key):
        # initialize the state
        # state are mutable data like the population, offsprings
        # the population is randomly initialized.
        # we don't have any offspring now, but initialize it as a placeholder
        # because jax want static shaped arrays.
        key, subkey = random.split(key)
        pop = random.uniform(subkey, (self.pop_size, self.ndim)) < 0.5
        return State(
            pop=pop,
            offsprings=jnp.empty((self.pop_size * 2, self.ndim)),
            fit=jnp.full((self.pop_size,), jnp.inf),
            key=key,
        )

    def ask(self, state):
        key, mut_key, x_key = random.split(state.key, 3)
        # here we do mutation and crossover (reproduction)
        # for simplicity, we didn't use any mating selections
        # so the offspring is twice as large as the population
        offsprings = jnp.concatenate(
            (
                mutation.bitflip(mut_key, state.pop, self.flip_prob),
                crossover.one_point(x_key, state.pop),
            ),
            axis=0,
        )
        # return the candidate solution and update the state
        return offsprings, state.update(offsprings=offsprings, key=key)

    def tell(self, state, fitness):
        # here we do selection
        merged_pop = jnp.concatenate([state.pop, state.offsprings])
        merged_fit = jnp.concatenate([state.fit, fitness])
        new_pop, new_fit = selection.topk_fit(merged_pop, merged_fit, self.pop_size)
        # replace the old population
        return state.update(pop=new_pop, fit=new_fit)
