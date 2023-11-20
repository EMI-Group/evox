# --------------------------------------------------------------------------------------
# 1. LMOCSO algorithm is described in the following papers:
#
# Title: Efficient Large-Scale Multiobjective Optimization Based on a Competitive Swarm
# Optimizer
# Link: https://ieeexplore.ieee.org/document/8681243

# 2. This code has been inspired by PlatEMO.
# More information about PlatEMO can be found at the following URL:
# GitHub Link: https://github.com/BIMK/PlatEMO
# --------------------------------------------------------------------------------------

import jax
import jax.numpy as jnp

from evox.operators import mutation, selection
from evox.operators.sampling import LatinHypercubeSampling
from evox import Algorithm, State, jit_class


@jax.jit
def cal_fitness(obj):
    # Calculate the fitness by shift-based density
    n = jnp.shape(obj)[0]
    f_max = jnp.max(obj, axis=0)
    f_min = jnp.min(obj, axis=0)
    f = (obj - jnp.tile(f_min, (n, 1))) / jnp.tile(f_max - f_min, (n, 1))

    s_obj = jax.vmap(lambda x: jnp.maximum(f, x))(f)

    def shifted_distance(f1, f2):
        s_f = jax.vmap(lambda x, y: jnp.linalg.norm(x - y), in_axes=(None, 0))(f1, f2)
        return s_f

    dis = jax.vmap(shifted_distance, in_axes=(0, 0))(f, s_obj)
    dis = jnp.where(jnp.eye(n), jnp.inf, dis)

    fitness = jnp.min(dis, axis=1)

    return fitness


@jit_class
class LMOCSO(Algorithm):
    """
    LMOCSO algorithm

    link: https://ieeexplore.ieee.org/document/8681243

    Args:
        alpha : The parameter controlling the rate of change of penalty. Defaults to 2.
        max_gen : The maximum number of generations. Defaults to 100.
        If the number of iterations is not 100, change the value based on the actual value.
    """

    def __init__(
        self,
        n_objs,
        lb,
        ub,
        pop_size,
        alpha=2,
        max_gen=100,
        selection_op=None,
        mutation_op=None,
    ):
        self.n_objs = n_objs
        self.dim = lb.shape[0]
        self.lb = lb
        self.ub = ub
        self.pop_size = pop_size
        self.alpha = alpha
        self.max_gen = max_gen

        self.selection = selection_op
        self.mutation = mutation_op

        if self.selection is None:
            self.selection = selection.ReferenceVectorGuided()
        if self.mutation is None:
            self.mutation = mutation.Polynomial((lb, ub))

        self.sampling = LatinHypercubeSampling(self.pop_size, self.n_objs)

    def setup(self, key):
        state_key, init_key, vector_key = jax.random.split(key, 3)

        population = (
            jax.random.uniform(init_key, shape=(self.pop_size, self.dim))
            * (self.ub - self.lb)
            + self.lb
        )
        velocity = jnp.zeros((self.pop_size, self.dim))
        fitness = jnp.full((self.pop_size, self.n_objs), jnp.inf)

        v = self.sampling(vector_key)[0]
        v = v / jnp.linalg.norm(v, axis=0)

        return State(
            population=population,
            next_generation=population,
            fitness=fitness,
            velocity=velocity,
            key=state_key,
            is_init=True,
            reference_vector=v,
            gen=0,
        )

    def ask(self, state):
        return jax.lax.cond(state.is_init, self._ask_init, self._ask_normal, state)

    def _ask_init(self, state):
        return state.population, state

    def _ask_normal(self, state):
        key, mating_key, pairing_key, r0_key, r1_key, mut_key = jax.random.split(
            state.key, 6
        )
        population = state.population
        no_nan_pop = ~jnp.isnan(population).all(axis=1)
        max_idx = jnp.sum(no_nan_pop).astype(int)
        pop = population[jnp.where(no_nan_pop, size=self.pop_size, fill_value=-1)]
        mating_pool = jax.random.randint(mating_key, (self.pop_size,), 0, max_idx)
        population = pop[mating_pool]

        randperm = jax.random.permutation(pairing_key, self.pop_size).reshape(2, -1)

        # calculate the shift-based density estimation(SDE) fitness
        sde_fitness = cal_fitness(state.fitness)

        mask = sde_fitness[randperm[0, :]] > sde_fitness[randperm[1, :]]

        winner = jnp.where(mask, randperm[0, :], randperm[1, :])
        loser = jnp.where(mask, randperm[1, :], randperm[0, :])

        r0 = jax.random.uniform(r0_key, shape=(self.pop_size // 2, self.dim))
        r1 = jax.random.uniform(r1_key, shape=(self.pop_size // 2, self.dim))

        off_velocity = r0 * state.velocity[loser] + r1 * (
            population[winner] - population[loser]
        )
        new_loser_population = jnp.clip(
            population[loser]
            + off_velocity
            + r0 * (off_velocity - state.velocity[loser]),
            self.lb,
            self.ub,
        )
        new_population = population.at[loser].set(new_loser_population)

        new_velocity = state.velocity.at[loser].set(off_velocity)

        next_generation = self.mutation(mut_key, new_population)

        return (
            next_generation,
            state.update(
                next_generation=next_generation,
                velocity=new_velocity,
                key=key,
            ),
        )

    def tell(self, state, fitness):
        return jax.lax.cond(
            state.is_init, self._tell_init, self._tell_normal, state, fitness
        )

    def _tell_init(self, state, fitness):
        state = state.update(fitness=fitness, is_init=False)
        return state

    def _tell_normal(self, state, fitness):
        current_gen = state.gen + 1
        v = state.reference_vector

        merged_pop = jnp.concatenate([state.population, state.next_generation])
        merged_fitness = jnp.concatenate([state.fitness, fitness])

        # RVEA Selection
        survivor, survivor_fitness = self.selection(
            merged_pop, merged_fitness, v, (current_gen / self.max_gen) ** self.alpha
        )

        state = state.update(
            population=survivor,
            fitness=survivor_fitness,
            gen=current_gen,
        )
        return state
