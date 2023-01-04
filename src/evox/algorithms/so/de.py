from functools import partial

import evox as ex
import jax
import jax.numpy as jnp
from jax import lax, vmap


@ex.jit_class
class DE(ex.Algorithm):
    def __init__(
        self,
        lb,
        ub,
        pop_size,
        base_vector="best",
        num_difference_vectors=1,
        cross_probability=0.9,
        differential_weight=0.8,
        batch_size=1,
        mean = None,
        stdvar = None
    ):
        assert jnp.all(lb < ub)
        assert pop_size >= 4
        assert cross_probability > 0 and cross_probability < 1
        assert pop_size % batch_size == 0
        assert base_vector in [
            "rand",
            "best",
        ], "base_vector must be either 'best' or 'rand'"

        self.num_difference_vectors = num_difference_vectors
        self.dim = lb.shape[0]
        self.lb = lb
        self.ub = ub
        self.pop_size = pop_size
        self.base_vector = base_vector
        self.batch_size = batch_size
        self.cross_probability = cross_probability
        self.differential_weight = differential_weight
        self.batch_size = batch_size
        self.mean = mean
        self.stdvar = stdvar

    def setup(self, key):
        state_key, init_key = jax.random.split(key)
        if self.mean is not None and self.stdvar is not None:
            population = self.stdvar * jax.random.normal(init_key, shape=(self.pop_size, self.dim))
            population = jnp.clip(population, self.lb, self.ub)
        else:
            population = jax.random.uniform(init_key, shape=(self.pop_size, self.dim))
            population = population * (self.ub - self.lb) + self.lb
        fitness = jnp.full((self.pop_size,), jnp.inf)
        best_index = 0
        start_index = 0

        return ex.State(
            population=population,
            fitness=fitness,
            best_index=best_index,
            start_index=start_index,
            key=state_key,
        )

    def ask(self, state):
        key, ask_one_key, R_key = jax.random.split(state.key, 3)
        R = jax.random.choice(R_key, self.dim, shape=(self.batch_size,))
        ask_one_keys = jax.random.split(ask_one_key, self.batch_size)
        indices = jnp.arange(self.batch_size) + state.start_index
        trial_vectors = vmap(
            partial(
                self._ask_one, population=state.population, best_index=state.best_index
            )
        )(ask_one_keys, indices, R)

        return state.update(trial_vectors=trial_vectors, key=key), trial_vectors

    def _ask_one(self, key, index, R, population, best_index):
        # index is the index of the "agent"
        # R is the dim that must change

        # first sample from [0, pop_size - 1), then replace best_index with pop_size - 1
        # this can ensure the same vector is not selected.
        select_key, crossover_key = jax.random.split(key)
        random_choiced = jax.random.choice(
            select_key,
            self.pop_size - 1,
            shape=(self.num_difference_vectors + 1,),
            replace=False,
        )
        random_choiced = jnp.where(
            random_choiced == index, self.pop_size - 1, random_choiced
        )

        if self.base_vector == "best":
            base_vector = population[best_index, :]
        else:
            base_vector = population[random_choiced[0], :]

        difference_vectors = population[random_choiced[1:], :]

        mutation_vectors = (
            jnp.sum(difference_vectors.at[1:, :].multiply(-1), axis=0)
            * self.differential_weight
            + base_vector
        )

        mask = (
            jax.random.uniform(crossover_key, shape=(self.dim,))
            < self.cross_probability
        )
        mask = mask.at[R].set(True)

        return jnp.where(
            mask,
            mutation_vectors,
            population[index],
        )

    def tell(self, state, trial_fitness):
        start_index = state.start_index
        batch_pop = jax.lax.dynamic_slice_in_dim(
            state.population, start_index, self.batch_size, axis=0
        )
        batch_fitness = jax.lax.dynamic_slice_in_dim(
            state.fitness, start_index, self.batch_size, axis=0
        )

        compare = trial_fitness < batch_fitness
        population_update = jnp.where(
            compare[:, jnp.newaxis], state.trial_vectors, batch_pop
        )
        fitness_update = jnp.where(compare, trial_fitness, batch_fitness)

        population = jax.lax.dynamic_update_slice_in_dim(
            state.population, population_update, start_index, axis=0
        )
        fitness = jax.lax.dynamic_update_slice_in_dim(
            state.fitness, fitness_update, start_index, axis=0
        )
        best_index = jnp.argmin(fitness)
        start_index = (state.start_index + self.batch_size) % self.pop_size
        return state.update(
            population=population,
            fitness=fitness,
            best_index=best_index,
            start_index=start_index,
        )
