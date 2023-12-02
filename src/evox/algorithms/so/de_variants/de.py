from functools import partial

import jax
import jax.numpy as jnp
from jax import vmap

from evox import Algorithm, State, jit_class


@jit_class
class DE(Algorithm):
    def __init__(
        self,
        lb,
        ub,
        pop_size,
        base_vector="rand",
        num_difference_vectors=1,
        differential_weight=0.5,
        cross_probability=0.9,
        batch_size=100,
        replace=False,
        mean=None,
        stdvar=None,
    ):
        assert jnp.all(lb < ub)
        assert pop_size >= 4
        assert cross_probability > 0 and cross_probability <= 1
        assert base_vector in [
            "rand",
            "best",
        ]

        self.num_difference_vectors = num_difference_vectors
        self.dim = lb.shape[0]
        self.lb = lb
        self.ub = ub
        self.pop_size = pop_size
        self.base_vector = base_vector
        self.batch_size = batch_size
        self.replace = replace
        self.cross_probability = cross_probability
        self.differential_weight = differential_weight
        self.mean = mean
        self.stdvar = stdvar

    def setup(self, key):
        state_key, init_key = jax.random.split(key)
        if self.mean is not None and self.stdvar is not None:
            population = self.stdvar * jax.random.normal(
                init_key, shape=(self.pop_size, self.dim)
            )
            population = jnp.clip(population, self.lb, self.ub)
        else:
            population = jax.random.uniform(init_key, shape=(self.pop_size, self.dim))
            population = population * (self.ub - self.lb) + self.lb
        fitness = jnp.full((self.pop_size,), jnp.inf)
        best_index = 0
        start_index = 0

        return State(
            population=population,
            fitness=fitness,
            best_index=best_index,
            start_index=start_index,
            key=state_key,
            trial_vectors=jnp.empty((self.batch_size, self.dim)),
        )

    def ask(self, state):
        key, R_key = jax.random.split(state.key, 2)

        indices = jnp.arange(self.batch_size) + state.start_index

        if self.replace:
            random_choices = jax.random.choice(
                R_key,
                self.pop_size,
                shape=(self.batch_size, self.num_difference_vectors * 2 + 1),
                replace=True,
            )
        else:
            choice_keys = jax.random.split(R_key, self.batch_size)
            random_choices = vmap(
                partial(
                    jax.random.choice,
                    a=self.pop_size,
                    shape=(self.num_difference_vectors * 2 + 1,),
                    replace=False,
                )
            )(choice_keys)

        R = jax.random.choice(R_key, self.dim, shape=(self.batch_size,))
        masks_init = (
            jax.random.uniform(R_key, shape=(self.batch_size, self.dim))
            < self.cross_probability
        )
        tile_arange = jnp.tile(jnp.arange(self.dim), (self.batch_size, 1))
        tile_R = jnp.tile(R[:, jnp.newaxis], (1, self.dim))
        masks = jnp.where(tile_arange == tile_R, True, masks_init)

        trial_vectors = vmap(
            partial(
                self._ask_one, population=state.population, best_index=state.best_index
            )
        )(indices, R, random_choiced=random_choices, mask=masks)

        return trial_vectors, state.update(trial_vectors=trial_vectors, key=key)

    def _ask_one(self, index, R, population, best_index, random_choiced, mask):
        random_choiced = jnp.where(
            random_choiced == index, self.pop_size - 1, random_choiced
        )

        if self.base_vector == "best":
            base_vector = population[best_index, :]
        else:
            base_vector = population[random_choiced[0], :]

        difference_vectors = population[random_choiced[1:], :]
        subtrahend_index = jnp.arange(1, self.num_difference_vectors * 2 + 1, 2)
        mutation_vectors = (
            jnp.sum(difference_vectors.at[subtrahend_index, :].multiply(-1), axis=0)
            * self.differential_weight
            + base_vector
        )

        trial_vector = jnp.where(
            mask,
            mutation_vectors,
            population[index],
        )
        trial_vector = jnp.clip(trial_vector, self.lb, self.ub)

        return trial_vector

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
