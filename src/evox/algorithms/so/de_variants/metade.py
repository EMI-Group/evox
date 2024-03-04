from evox import (
    Problem,
    Algorithm,
    State,
    jit_class,
)
from jax import vmap, lax
from evox.operators.selection import select_rand_pbest
from evox.operators.crossover import (
    de_diff_sum,
    de_arith_recom,
    de_bin_cross,
    de_exp_cross,
)
from evox.utils import *


# MetaDE inherits from the problem class and has a different computational flow from the above algorithms.
# See test_single_objective_algorithms.py for the flow

# A decoder function for DE that maps population vectors to algorithm parameters.
def decoder_de(pop):
    return {
        "differential_weight": pop[:, 0],  # Weight for differential evolution.
        "cross_probability": pop[:, 1],  # Probability of crossover.
        "basevect_prim_type": jnp.floor(pop[:, 2]).astype(int),  # Primary base vector type.
        "basevect_sec_type": jnp.floor(pop[:, 3]).astype(int),  # Secondary base vector type.
        "num_diff_vects": jnp.floor(pop[:, 4]).astype(int),  # Number of difference vectors.
        "cross_strategy": jnp.floor(pop[:, 5]).astype(int),  # Crossover strategy.
    }


# A function to create a batch algorithm class from a base algorithm.
def create_batch_algorithm(base_algorithm, batch_size, num_runs):
    @jit_class
    class BatchAlgorithm(base_algorithm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.batch_size = batch_size
            self.num_runs = num_runs

        def setup(self, key):
            # Set up the algorithm state for each run in the batch.
            state = vmap(vmap(super().setup))(
                jnp.broadcast_to(jax.random.split(key, num=self.num_runs),
                                 (self.batch_size, self.num_runs, 2))
            )
            return state

        def ask(self, state):
            # Generate new solutions to evaluate.
            return vmap(vmap(super().ask))(state)

        def tell(self, state, fitness):
            # Update the algorithm state with new fitness information.
            return vmap(vmap(super().tell))(state, fitness)

        def override(self, state, key, params):
            # Override the parameters of the base algorithms.
            return vmap(
                vmap(super().override, in_axes=(0, 0, None)),
                in_axes=(0, None, 0),
            )(state, jax.random.split(key, num=self.num_runs), params)

    return BatchAlgorithm


@jit_class
class MetaDE(Problem):
    # MetaDE class for optimizing the hyperparameters of a base algorithm on a given problem.
    def __init__(
            self,
            base_algorithm,
            problem,
            batch_size,
            num_runs,
            base_alg_steps,
            override=True
    ):
        super().__init__()
        self.base_algorithm = base_algorithm
        self.problem = problem
        self.batch_size = batch_size
        self.num_runs = num_runs
        self.override = override
        self.base_alg_steps = base_alg_steps

    def setup(self, key):
        # Setup the MetaDE problem.
        return State(key=key)

    def evaluate(self, state, x):
        # Evaluate the MetaDE problem with a given set of parameters x.
        override_key, _ = jax.random.split(state.key, num=2)
        if self.override:
            state = self.base_algorithm.override(state, override_key, x)

        # Evaluate the base problem using the overridden parameters.
        evaluate = vmap(
            vmap(self.problem.evaluate, in_axes=(None, 0), out_axes=(0, None)),
            in_axes=(None, 0),
            out_axes=(0, None),
        )

        def one_step(_i, min_fit_and_pso_state):
            # One optimization step.
            min_fitness, state = min_fit_and_pso_state
            pops, state = self.base_algorithm.ask(state)
            fitness, state = evaluate(state, pops)
            state = self.base_algorithm.tell(state, fitness)
            min_fitness = jnp.minimum(
                min_fitness, jnp.nanmin(jnp.nanmin(fitness, axis=2), axis=1)
            )
            return min_fitness, state

        # Determine the number of base algorithm steps, potentially increasing them if `power_up` is set.
        base_alg_steps = jax.lax.select(state.power_up, self.base_alg_steps * 5, self.base_alg_steps)

        # Run the base algorithm for a specified number of steps.
        min_fitness, state = jax.lax.fori_loop(
            0, base_alg_steps, one_step, (jnp.full((self.batch_size,), jnp.inf), state)
        )

        return min_fitness, state


@jit_class
class ParamDE(Algorithm):
    """Parametric DE class."""

    def __init__(
            self,
            lb,
            ub,
            pop_size=100,
            diff_padding_num=9,
            differential_weight=0.3471559,
            cross_probability=0.78762645,
            basevect_prim_type=0,
            basevect_sec_type=2,
            num_diff_vects=3,
            cross_strategy=2,
    ):
        # Initialize the ParamDE algorithm with given parameters.
        self.num_diff_vects = num_diff_vects
        self.dim = lb.shape[0]
        self.lb = lb
        self.ub = ub
        self.pop_size = pop_size
        self.batch_size = pop_size
        self.cross_probability = cross_probability
        self.differential_weight = differential_weight
        self.cross_strategy = cross_strategy
        self.diff_padding_num = diff_padding_num
        self.basevect_prim_type = basevect_prim_type
        self.basevect_sec_type = basevect_sec_type

    def setup(self, key):
        # Setup the ParamDE algorithm state.
        state_key, init_key = jax.random.split(key, 2)
        population = jax.random.uniform(init_key, shape=(self.pop_size, self.dim))
        population = population * (self.ub - self.lb) + self.lb
        fitness = jnp.full((self.pop_size,), jnp.inf)
        trial_vectors = jnp.zeros(shape=(self.batch_size, self.dim))
        best_index = 0
        start_index = 0
        params_init = {
            "differential_weight": self.differential_weight,
            "cross_probability": self.cross_probability,
            "basevect_prim_type": self.basevect_prim_type,
            "basevect_sec_type": self.basevect_sec_type,
            "num_diff_vects": self.num_diff_vects,
            "cross_strategy": self.cross_strategy,
        }
        return State(
            population=population,
            fitness=fitness,
            best_index=best_index,
            start_index=start_index,
            key=state_key,
            trial_vectors=trial_vectors,
            params=params_init,
        )

    def ask(self, state):
        # Generate new candidate solutions.
        key, ask_one_key = jax.random.split(state.key, 2)
        ask_one_keys = jax.random.split(ask_one_key, self.batch_size)
        indices = jnp.arange(self.batch_size) + state.start_index

        trial_vectors = vmap(partial(self._ask_one, state_inner=state))(ask_one_key=ask_one_keys, index=indices)

        return trial_vectors, state.update(trial_vectors=trial_vectors, key=key)

    def _ask_one(self, state_inner, ask_one_key, index):
        # Generate a single candidate solution.
        select_key, pbest_key, crossover_key = jax.random.split(ask_one_key, 3)

        population = state_inner.population
        best_index = state_inner.best_index
        fitness = state_inner.fitness
        params = state_inner.params

        # Mutation and crossover operations.
        difference_sum, rand_vect_idx = de_diff_sum(
            select_key,
            self.diff_padding_num,
            params["num_diff_vects"],
            index,
            population,
        )

        rand_vect = population[rand_vect_idx]
        best_vect = population[best_index]
        pbest_vect = select_rand_pbest(pbest_key, 0.05, population, fitness)
        current_vect = population[index]
        vector_merge = jnp.stack((rand_vect, best_vect, pbest_vect, current_vect))

        base_vector_prim = vector_merge[params["basevect_prim_type"]]
        base_vector_sec = vector_merge[params["basevect_sec_type"]]

        base_vector = base_vector_prim + params["differential_weight"] * (base_vector_sec - base_vector_prim)

        mutation_vector = (base_vector + difference_sum * params["differential_weight"])

        # Select the crossover strategy and perform crossover.
        cross_funcs = (
            de_bin_cross,
            de_exp_cross,
            lambda _key, x, y, z: de_arith_recom(x, y, z),
        )
        trial_vector = lax.switch(
            params["cross_strategy"],
            cross_funcs,
            crossover_key,
            mutation_vector,
            current_vect,
            params["cross_probability"],
        )

        trial_vector = jnp.clip(trial_vector, self.lb, self.ub)

        return trial_vector

    def tell(self, state, trial_fitness):
        # Update the algorithm state with the fitness of the candidate solutions.
        start_index = state.start_index
        batch_pop = jax.lax.dynamic_slice_in_dim(
            state.population, start_index, self.batch_size, axis=0
        )
        batch_fitness = jax.lax.dynamic_slice_in_dim(
            state.fitness, start_index, self.batch_size, axis=0
        )

        compare = trial_fitness <= batch_fitness

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

    def override(self, state, key, params):
        # Override the algorithm parameters with new values.
        state = state | ParamDE.setup(self, key)
        return state.update(params=params)
