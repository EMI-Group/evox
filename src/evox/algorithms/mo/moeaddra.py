# --------------------------------------------------------------------------------------
# 1. MOEA/D-DRA algorithm is described in the following papers:
#
# Title: The performance of a new version of MOEA/D on CEC09 unconstrained MOP test instances
# Link: https://ieeexplore.ieee.org/abstract/document/4982949
#
# 2. This code has been inspired by PlatEMO.
# More information about PlatEMO can be found at the following URL:
# GitHub Link: https://github.com/BIMK/PlatEMO
# --------------------------------------------------------------------------------------

import jax
import jax.numpy as jnp

from evox import jit_class, Algorithm, State
from evox.operators import selection, mutation, crossover
from evox.operators.sampling import UniformSampling, LatinHypercubeSampling
from evox.utils import pairwise_euclidean_dist


@jit_class
class MOEADDRA(Algorithm):
    """MOEA/D-DRA algorithm

    link: https://ieeexplore.ieee.org/abstract/document/4982949
    """
    
    def __init__(
        self,
        lb,
        ub,
        n_objs,
        pop_size,
        # type=1,
        mutation_op=None,
        crossover_op=None,
    ):
        self.lb = lb
        self.ub = ub
        self.n_objs = n_objs
        self.dim = lb.shape[0]
        self.pop_size = pop_size
        self.type = type
        self.T = jnp.ceil(self.pop_size / 10).astype(int)
        self.nr = jnp.ceil(self.pop_size / 100).astype(int)
        self.i_size = jnp.floor(self.pop_size / 5).astype(int)

        self.mutation = mutation_op
        self.crossover = crossover_op

        self.selection = selection.Tournament(n_round=self.pop_size)
        if self.mutation is None:
            self.mutation = mutation.Polynomial((lb, ub))
        if self.crossover is None:
            self.crossover = crossover.DifferentialEvolve()
        self.sample = LatinHypercubeSampling(self.pop_size, self.n_objs)

    def setup(self, key):
        key, subkey1, subkey2 = jax.random.split(key, 3)
        population = (
            jax.random.uniform(subkey1, shape=(self.pop_size, self.dim))
            * (self.ub - self.lb)
            + self.lb
        )

        w = self.sample(subkey2)[0]
        B = pairwise_euclidean_dist(w, w)
        B = jnp.argsort(B, axis=1)
        B = B[:, : self.T]
        return State(
            population=population,
            fitness=jnp.zeros((self.pop_size, self.n_objs)),
            next_generation=population,
            weight_vector=w,
            B=B,
            Z=jnp.zeros(shape=self.n_objs),
            pi=jnp.ones((self.pop_size,)),
            old_obj=jnp.zeros((self.pop_size,)),
            choosed_p=jnp.zeros((self.pop_size, self.T)).astype(int),
            I_all=jnp.zeros((self.pop_size,)).astype(int),
            gen=0,
            is_init=True,
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

        key, subkey1, subkey2, subkey3, sel_key, x_key, mut_key = jax.random.split(
            state.key, 7
        )
        parent = jax.random.permutation(
            subkey1, state.B, axis=1, independent=True
        ).astype(int)
        rand = jax.random.uniform(subkey2, (self.pop_size, 1))
        rand_perm = jax.random.randint(
            subkey3, (self.pop_size, self.T), 0, self.pop_size
        )
        w = state.weight_vector
        pi = state.pi
        population = state.population

        _, selected_idx = self.selection(sel_key, population, -pi)
        mask = jnp.sum(w < 1e-3, axis=1) == (self.n_objs - 1)

        boundary = jnp.where(mask, size=self.pop_size, fill_value=0)[0]
        boundary = boundary[: self.i_size]
        g_bound = jnp.tile(boundary, (5,))

        I_all = jnp.where(g_bound != 0, g_bound, selected_idx)

        choosed_p = jnp.where(rand < 0.9, parent[I_all], rand_perm)

        crossovered = self.crossover(
            x_key,
            population[I_all],
            population[choosed_p[:, 0]],
            population[choosed_p[:, 1]],
        )
        next_generation = self.mutation(mut_key, crossovered)

        return next_generation, state.update(
            next_generation=next_generation, choosed_p=choosed_p, key=key, I_all=I_all
        )

    def _tell_init(self, state, fitness):
        Z = jnp.min(fitness, axis=0)
        old_obj = jnp.max(
            jnp.abs((fitness - jnp.tile(Z, (self.pop_size, 1))) * state.weight_vector),
            axis=1,
        )
        state = state.update(fitness=fitness, Z=Z, old_obj=old_obj, is_init=False)
        return state

    def _tell_normal(self, state, fitness):
        current_gen = state.gen + 1
        population = state.population
        pop_obj = state.fitness
        offspring = state.next_generation
        off_obj = fitness
        w = state.weight_vector
        Z = state.Z
        choosed_p = state.choosed_p
        pi = state.pi
        old_obj = state.old_obj

        out_vals = (population, pop_obj, Z)

        def out_body(i, out_vals):
            population, pop_obj, Z = out_vals
            p = choosed_p[i]

            ind_dec = offspring[i]
            ind_obj = off_obj[i]
            Z = jnp.minimum(Z, ind_obj)

            g_old = jnp.max(
                jnp.abs(pop_obj[p] - jnp.tile(Z, (len(p), 1))) * w[p], axis=1
            )
            g_new = jnp.max(jnp.abs(jnp.tile(ind_obj - Z, (len(p), 1))) * w[p], axis=1)

            g_new = g_new[:, jnp.newaxis]
            g_old = g_old[:, jnp.newaxis]

            indices = jnp.where(g_old >= g_new, size=len(p))[0][: self.nr]

            population = population.at[p[indices]].set(ind_dec)
            pop_obj = pop_obj.at[p[indices]].set(ind_obj)

            return (population, pop_obj, Z)

        population, pop_obj, Z = jax.lax.fori_loop(0, self.pop_size, out_body, out_vals)

        def update_pi(pi, old_obj):
            new_obj = jnp.max(
                jnp.abs((pop_obj - jnp.tile(Z, (self.pop_size, 1))) * w), axis=1
            )
            delta = (old_obj - new_obj) / old_obj
            mask = delta < 0.001
            pi = jnp.where(mask, pi * (0.95 + 0.05 * delta / 0.001), 1)
            old_obj = new_obj
            return pi, old_obj

        def no_update(pi, old_obj):
            return pi, old_obj

        pi, old_obj = jax.lax.cond(
            current_gen % 10 == 0,
            update_pi,
            no_update,
            pi,
            old_obj,
        )

        state = state.update(
            population=population,
            fitness=pop_obj,
            Z=Z,
            gen=current_gen,
            pi=pi,
            old_obj=old_obj,
        )
        return state
