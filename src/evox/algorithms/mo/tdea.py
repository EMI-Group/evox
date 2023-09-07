# --------------------------------------------------------------------------------------
# 1. Theta-dominance based evolutionary algorithm is described in the following papers:
#
# Title: A New Dominance Relation-Based Evolutionary Algorithm for Many-Objective Optimization
# Link: https://ieeexplore.ieee.org/abstract/document/7080938
#
# 2. This code has been inspired by PlatEMO.
# More information about PlatEMO can be found at the following URL:
# GitHub Link: https://github.com/BIMK/PlatEMO
# --------------------------------------------------------------------------------------

import jax
import jax.numpy as jnp

from evox.operators import mutation, crossover, sampling
from evox.operators import non_dominated_sort
from evox.utils import cos_dist
from evox import Algorithm, State, jit_class
from functools import partial


@jax.jit
def theta_nd_sort(obj, w, mask):
    # theta non-dominated sort
    n = len(obj)
    nw = len(w)

    norm_p = jnp.sqrt(jnp.sum(obj**2, axis=1, keepdims=True))
    cosine = cos_dist(obj, w)
    cosine = jnp.clip(cosine, -1, 1)
    d1 = jnp.tile(norm_p, (1, nw)) * cosine
    d2 = jnp.tile(norm_p, (1, nw)) * jnp.sqrt(1 - cosine**2)

    d_class = jnp.argmin(d2, axis=1)
    d_class = jnp.where(mask, d_class, jnp.inf)

    theta = jnp.zeros((nw,)) + 5
    theta = jnp.where(jnp.sum(w > 1e-4, axis=1) == 1, 1e6, theta)
    t_rank = jnp.zeros((n,), dtype=int)

    t_list = jnp.arange(1, n + 1)

    def loop_body(i, val):
        t_front_no = val
        loop_mask = d_class == i
        d = jnp.where(loop_mask, d1[:, i] + theta[i] * d2[:, i], jnp.inf)
        rank = jnp.argsort(d)
        tmp = jnp.where(loop_mask[rank], t_list, t_front_no[rank])
        t_front_no = t_front_no.at[rank].set(tmp)

        return t_front_no

    t_rank = jax.lax.fori_loop(0, nw, loop_body, t_rank)
    t_rank = jnp.where(mask, t_rank, jnp.inf)

    return t_rank


@partial(jax.jit, static_argnums=3)
def environmental_selection(pop, obj, w, n, z, z_nad):

    n_merge, m = jnp.shape(obj)
    rank = non_dominated_sort(obj)
    order = jnp.argsort(rank)
    worst_rank = rank[order[n]]
    mask = rank <= worst_rank

    z = jnp.minimum(z, jnp.min(obj, axis=0))

    w1 = jnp.zeros((m, m)) + 1e-6
    w1 = jnp.where(jnp.eye(m), 1, w1)
    asf = jax.vmap(
        lambda x, y: jnp.max(jnp.abs((x - z) / (z_nad - z)) / y, axis=1),
        in_axes=(None, 0),
        out_axes=1,
    )(obj, w1)

    extreme = jnp.argmin(asf, axis=0)
    hyper_plane = jnp.linalg.solve(
        obj[extreme, :] - jnp.tile(z, (m, 1)), jnp.ones((m, 1))
    )
    a = z + 1 / jnp.squeeze(hyper_plane)

    a = jax.lax.cond(
        jnp.any(jnp.isnan(a)) | jnp.any(a <= z),
        lambda _: jnp.max(obj, axis=0),
        lambda x: x,
        a,
    )
    z_nad = a

    norm_obj = (obj - jnp.tile(z, (n_merge, 1))) / jnp.tile(z_nad - z, (n_merge, 1))

    t_rank = theta_nd_sort(norm_obj, w, mask)
    combined_order = jnp.lexsort((t_rank, rank))[:n]

    return pop[combined_order], obj[combined_order], z, z_nad


@jit_class
class TDEA(Algorithm):
    """Theta-dominance based evolutionary algorithm

    link: https://ieeexplore.ieee.org/abstract/document/7080938
    """

    def __init__(
        self,
        lb,
        ub,
        n_objs,
        pop_size,
        mutation_op=None,
        crossover_op=None,
    ):
        self.lb = lb
        self.ub = ub
        self.n_objs = n_objs
        self.dim = lb.shape[0]
        self.pop_size = pop_size

        self.mutation = mutation_op
        self.crossover = crossover_op

        if self.mutation is None:
            self.mutation = mutation.Polynomial((lb, ub))
        if self.crossover is None:
            self.crossover = crossover.SimulatedBinary()
        self.sampling = sampling.LatinHypercubeSampling(self.pop_size, self.n_objs)

    def setup(self, key):
        key, subkey1, subkey2 = jax.random.split(key, 3)
        population = (
            jax.random.uniform(subkey1, shape=(self.pop_size, self.dim))
            * (self.ub - self.lb)
            + self.lb
        )
        w = self.sampling(subkey2)[0]
        return State(
            population=population,
            fitness=jnp.zeros((self.pop_size, self.n_objs)),
            next_generation=population,
            w=w,
            z=jnp.zeros((self.n_objs,)),
            z_nad=jnp.zeros((self.n_objs,)),
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
        key, sel_key, x_key, mut_key = jax.random.split(state.key, 4)

        mating_pool = jax.random.randint(sel_key, (self.pop_size,), 0, self.pop_size)
        population = state.population[mating_pool]
        crossovered = self.crossover(x_key, population)
        next_generation = self.mutation(mut_key, crossovered)

        return next_generation, state.update(next_generation=next_generation, key=key)

    def _tell_init(self, state, fitness):
        z = jnp.min(fitness, axis=0)
        z_nad = jnp.max(fitness, axis=0)
        state = state.update(fitness=fitness, z=z, z_nad=z_nad, is_init=False)
        return state

    def _tell_normal(self, state, fitness):
        merged_pop = jnp.concatenate([state.population, state.next_generation], axis=0)
        merged_fitness = jnp.concatenate([state.fitness, fitness], axis=0)

        population, pop_obj, z, z_nad = environmental_selection(
            merged_pop, merged_fitness, state.w, self.pop_size, state.z, state.z_nad
        )

        state = state.update(population=population, fitness=pop_obj, z=z, z_nad=z_nad)
        return state
