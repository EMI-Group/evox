# --------------------------------------------------------------------------------------
# 1. Bi-criterion evolution based IBEA algorithm is described in the following papers:
#
# Title: Pareto or Non-Pareto: Bi-Criterion Evolution in Multiobjective Optimization
# Link: https://ieeexplore.ieee.org/abstract/document/7347391
#
# 2. This code has been inspired by PlatEMO.
# More information about PlatEMO can be found at the following URL:
# GitHub Link: https://github.com/BIMK/PlatEMO
# --------------------------------------------------------------------------------------

import jax
import jax.numpy as jnp

from evox.operators import (
    non_dominated_sort,
    selection,
    mutation,
    crossover,
)
from evox import Algorithm, jit_class, State
from evox.utils import cal_max, pairwise_euclidean_dist
from functools import partial


@jax.jit
def cal_fitness(pop_obj, kappa):
    n = jnp.shape(pop_obj)[0]
    pop_obj = (pop_obj - jnp.tile(jnp.min(pop_obj), (n, 1))) / (
        jnp.tile(jnp.max(pop_obj) - jnp.min(pop_obj), (n, 1))
    )
    I = cal_max(pop_obj, pop_obj)

    C = jnp.max(jnp.abs(I), axis=0)

    fitness = jnp.sum(-jnp.exp(-I / jnp.tile(C, (n, 1)) / kappa), axis=0) + 1

    return fitness, I, C


@partial(jax.jit, static_argnums=3)
def exploration(pc_obj, npc_obj, n_nd, n):
    """
    Pareto criterion evolving

    Args:
        pc_obj: Objective values of Pareto criterion solutions.
        npc_obj: Objective values of non-Pareto criterion solutions.
        n_nd: Number of nondominated solutions.
        n: Total number of solutions.

    Returns:
        s: Boolean array indicating solutions to be explored.
    """
    f_max = jnp.max(pc_obj, axis=0)
    f_min = jnp.min(pc_obj, axis=0)
    norm_pc_obj = (pc_obj - jnp.tile(f_min, (len(pc_obj), 1))) / jnp.tile(
        f_max - f_min, (len(pc_obj), 1)
    )
    norm_npc_obj = (npc_obj - jnp.tile(f_min, (len(npc_obj), 1))) / jnp.tile(
        f_max - f_min, (len(npc_obj), 1)
    )

    # Determine the size of the niche
    distance = pairwise_euclidean_dist(norm_pc_obj, norm_pc_obj)
    distance = distance.at[
        jnp.arange(0, len(norm_pc_obj)), jnp.arange(0, len(norm_pc_obj))
    ].set(jnp.inf)
    distance = jnp.where(jnp.isnan(distance), jnp.inf, distance)
    distance = jnp.sort(distance, axis=1)
    # Calculate the characteristic distance r0 for niche detection
    r0 = jnp.mean(distance[:, jnp.minimum(2, jnp.shape(distance)[1] - 1)])
    r = n_nd / n * r0

    # Detect the solutions in PC to be explored
    # s: Solutions to be explored
    distance = pairwise_euclidean_dist(norm_pc_obj, norm_npc_obj)
    s = jnp.sum(distance <= r, axis=1) <= 1

    return s


@partial(jax.jit, static_argnums=2)
def pc_selection(pc, pc_obj, n):
    """
    Pareto criterion selection

    Args:
        pc : Pareto criterion population.
        pc_obj : Objective values of pc.
        n : Number of solutions to select.
    """
    # m: Number of objectives
    # n_nd: Number of non-dominated solutions in PC
    m = jnp.shape(pc_obj)[1]
    rank = non_dominated_sort(pc_obj)
    mask = rank == 0
    n_nd = jnp.sum(mask).astype(int)
    mask = mask[:, jnp.newaxis]
    next_ind = jnp.zeros(n, dtype=jnp.int32)
    i = n_nd

    def true_fun(next_ind, mask, i):
        f_max = jnp.max(jnp.where(jnp.tile(mask, (1, m)), pc_obj, -jnp.inf), axis=0)
        f_min = jnp.min(jnp.where(jnp.tile(mask, (1, m)), pc_obj, jnp.inf), axis=0)
        norm_obj = (pc_obj - jnp.tile(f_min, (len(pc_obj), 1))) / jnp.tile(
            f_max - f_min, (len(pc_obj), 1)
        )
        norm_obj = jnp.where(jnp.tile(mask, (1, m)), norm_obj, jnp.inf)
        distance = pairwise_euclidean_dist(norm_obj, norm_obj)
        distance = distance.at[
            jnp.arange(0, len(norm_obj)), jnp.arange(0, len(norm_obj))
        ].set(jnp.inf)
        distance = jnp.where(jnp.isnan(distance), jnp.inf, distance)

        # Calculate sorted distance matrix (sd) for each solution
        sd = jnp.sort(distance, axis=1)
        sd = jnp.where(jnp.tile(mask, (1, len(pc_obj))), sd, 0)

        # Calculate the characteristic distance r for niche detection
        r = jnp.sum(sd[:, jnp.minimum(2, jnp.shape(sd)[1] - 1)]) / n_nd

        # Calculate big_r which scales the distance matrix
        big_r = jnp.minimum(distance / r, 1)

        def loop(vals):
            i, mask, big_r = vals
            idx = jnp.argmax(1 - jnp.prod(big_r, axis=0))
            mask = mask.at[idx].set(False)
            big_r = big_r.at[idx, :].set(1)
            big_r = big_r.at[:, idx].set(1)
            return (i - 1, mask, big_r)

        _, mask, big_r = jax.lax.while_loop(lambda x: x[0] > n, loop, (i, mask, big_r))
        pc_indices = jnp.where(mask, size=len(mask), fill_value=-1)[0]
        next_ind = pc_indices[:n]
        return next_ind, mask, i

    def false_fun(next_ind, mask, i):
        pc_indices = jnp.where(mask, size=len(mask), fill_value=-1)[0]
        head = pc_indices[0]
        pc_indices = jnp.where(pc_indices == -1, head, pc_indices)
        next_ind = pc_indices[:n]
        return next_ind, mask, i

    next_ind, _, _ = jax.lax.cond(n_nd > n, true_fun, false_fun, next_ind, mask, i)

    return pc[next_ind], pc_obj[next_ind], n_nd


@partial(jax.jit, static_argnums=2)
def environmental_selection(pop, obj, n, kappa):

    merged_fitness, I, C = cal_fitness(obj, kappa)
    next_ind = jnp.arange(len(pop))
    vals = (next_ind, merged_fitness)

    def body_fun(i, vals):
        next_ind, merged_fitness = vals
        x = jnp.argmin(merged_fitness)
        merged_fitness += jnp.exp(-I[x, :] / C[x] / kappa)
        merged_fitness = merged_fitness.at[x].set(jnp.max(merged_fitness))
        next_ind = next_ind.at[x].set(-1)
        return (next_ind, merged_fitness)

    next_ind, merged_fitness = jax.lax.fori_loop(0, n, body_fun, vals)

    ind = jnp.where(next_ind != -1, size=len(pop), fill_value=-1)[0]
    ind_n = ind[0:n]

    return pop[ind_n], obj[ind_n]


@jit_class
class BCEIBEA(Algorithm):
    """Bi-criterion evolution based IBEA

    link: https://ieeexplore.ieee.org/abstract/document/7347391

    Note: The number of outer iterations needs to be set to Maximum Generation*2+1.

    Args:
        kappa (float, optional): The scaling factor for selecting parents in the environmental selection.
            It controls the probability of selecting parents based on their fitness values.
            Defaults to 0.05.
    """

    def __init__(
        self,
        lb,
        ub,
        n_objs,
        pop_size,
        kappa=0.05,
        selection_op=None,
        mutation_op=None,
        crossover_op=None,
    ):
        self.lb = lb
        self.ub = ub
        self.n_objs = n_objs
        self.dim = lb.shape[0]
        self.pop_size = pop_size
        self.kappa = kappa

        self.selection = selection_op
        self.mutation = mutation_op
        self.crossover = crossover_op

        self.selection = selection.Tournament(n_round=self.pop_size)
        if self.mutation is None:
            self.mutation = mutation.Polynomial((self.lb, self.ub))
        if self.crossover is None:
            self.crossover = crossover.SimulatedBinary()
        self.crossover_odd = crossover.SimulatedBinary(type=2)

    def setup(self, key):
        key, subkey = jax.random.split(key)
        population = (
            jax.random.uniform(subkey, shape=(self.pop_size, self.dim))
            * (self.ub - self.lb)
            + self.lb
        )
        return State(
            population=population,
            fitness=jnp.zeros((self.pop_size, self.n_objs)),
            npc=population,
            npc_obj=jnp.zeros((self.pop_size, self.n_objs)),
            new_pc=population,
            new_pc_obj=jnp.zeros((self.pop_size, self.n_objs)),
            new_npc=population,
            new_npc_obj=jnp.zeros((self.pop_size, self.n_objs)),
            n_nd=0,
            next_generation=population,
            is_init=True,
            counter=1,
            key=key,
        )

    def ask(self, state):
        return jax.lax.cond(state.is_init, self._ask_init, self._ask_normal, state)

    def tell(self, state, fitness):
        return jax.lax.cond(
            state.is_init, self._tell_init, self._tell_normal, state, fitness
        )

    def _ask_init(self, state):
        return state.population, state

    def _ask_normal(self, state):
        return jax.lax.cond(
            state.counter % 2 == 0, self._ask_even, self._ask_odd, state
        )

    def _ask_odd(self, state):
        key, mating_key, x_key, mut_key = jax.random.split(state.key, 4)
        s = exploration(state.fitness, state.npc_obj, state.n_nd, self.pop_size)
        mating_pool = jax.random.randint(
            mating_key, shape=(self.pop_size,), minval=0, maxval=self.pop_size
        )
        head = jnp.where(s, size=len(s), fill_value=-1)[0]
        mating_pool = jnp.where(s, mating_pool, head[0])
        s_indices = jnp.where(s, jnp.arange(0, len(s)), head[0])
        pop = state.population
        indices = jnp.concatenate((s_indices, mating_pool))

        def true_fun(pop):
            mating_pop = pop[indices]
            coreeovered = self.crossover_odd(x_key, mating_pop)
            offspring = self.mutation(mut_key, coreeovered)
            return offspring

        pop = jax.lax.cond(jnp.sum(s) != 0, true_fun, lambda x: x, pop)

        return pop, state.update(new_pc=pop, key=key)

    def _ask_even(self, state):
        key, sel_key, x_key, mut_key = jax.random.split(state.key, 4)
        fit = -cal_fitness(state.npc_obj, self.kappa)[0]
        selected, _ = self.selection(sel_key, state.npc, fit)
        crossovered = self.crossover(x_key, selected)
        next_generation = self.mutation(mut_key, crossovered)

        return next_generation, state.update(new_npc=next_generation, key=key)

    def _tell_init(self, state, fitness):
        pc, pc_obj, n_nd = pc_selection(state.population, fitness, self.pop_size)
        state = state.update(
            population=pc,
            fitness=pc_obj,
            npc_obj=fitness,
            new_pc_obj=fitness,
            new_npc_obj=fitness,
            n_nd=n_nd,
            is_init=False,
        )
        return state

    def _tell_normal(self, state, fitness):
        return jax.lax.cond(
            state.counter % 2 == 0, self._tell_even, self._tell_odd, state, fitness
        )

    def _tell_odd(self, state, fitness):
        new_pc_obj = fitness
        merged_pop = jnp.concatenate([state.npc, state.new_pc], axis=0)
        merged_fitness = jnp.concatenate([state.npc_obj, fitness], axis=0)
        npc, npc_obj = environmental_selection(
            merged_pop, merged_fitness, self.pop_size, self.kappa
        )
        return state.update(
            npc=npc, fitness=npc_obj, counter=state.counter + 1, new_pc_obj=new_pc_obj
        )

    def _tell_even(self, state, fitness):
        new_npc_obj = fitness
        merged_pop = jnp.concatenate([state.npc, state.new_npc], axis=0)
        merged_fitness = jnp.concatenate([state.npc_obj, new_npc_obj], axis=0)

        npc, npc_obj = environmental_selection(
            merged_pop, merged_fitness, self.pop_size, self.kappa
        )

        merged_pop = jnp.concatenate(
            (state.population, state.new_npc, state.new_pc), axis=0
        )
        merged_fitness = jnp.concatenate(
            (state.npc_obj, new_npc_obj, state.new_pc_obj), axis=0
        )

        pc, pc_obj, n_nd = pc_selection(merged_pop, merged_fitness, self.pop_size)

        state = state.update(
            population=pc,
            fitness=pc_obj,
            n_nd=n_nd,
            new_npc_obj=new_npc_obj,
            npc=npc,
            npc_obj=npc_obj,
            counter=state.counter + 1,
        )
        return state
