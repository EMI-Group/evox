import jax
import jax.numpy as jnp

from evox import jit_class, Algorithm, State
from evox.operators.mutation import PmMutation
from evox.operators.crossover import SimulatedBinaryCrossover
from evox.operators.sampling import UniformSampling, LatinHypercubeSampling
from evox.utils import cos_dist
from jax.experimental.host_callback import id_print
from functools import partial


@partial(jax.jit, static_argnums=3)
def associate(pop, obj, w, s, rng):
    k = len(w)

    dis = cos_dist(obj, w)
    max_indices = jnp.argmax(dis, axis=1)
    id_print(max_indices)
    partition = jnp.zeros((s, k), dtype=int)

    def body_fun(i, p):
        mask = max_indices == i
        # current = jnp.zeros((s, ))
        current = jnp.where(mask, size=len(pop), fill_value=-1)[0]

        def true_fun(c):
            c = c[:s]
            rad = jax.random.randint(rng, (s, ), 0, len(pop))
            c = jnp.where(c != -1, c, rad)
            return c

        def false_fun(c):
            rank = non_dominated_sort(obj)
            rank = jnp.where(mask, rank, jnp.inf)
            order = jnp.argsort(rank)
            worst_rank = rank[order[s - 1]]
            mask_worst = rank == worst_rank
            crowding_dis = crowding_distance(obj, mask_worst)
            c = jnp.lexsort((-crowding_dis, rank))[: s]
            return c

        current = jax.lax.cond(jnp.sum(mask) < s, true_fun, false_fun, current)
        p = p.at[:, i].set(current)
        return p

    partition = jax.lax.fori_loop(0, k, body_fun, partition)
    id_print(partition)
    partition = partition.flatten(order='F')

    return partition


@jit_class
class MOEADM2M(Algorithm):
    """MOEA/D based on MOP to MOP algorithm

    link: https://ieeexplore.ieee.org/abstract/document/6595549
    """

    def __init__(
        self,
        lb,
        ub,
        n_objs,
        pop_size,
        k=10,
        mutation=PmMutation(),
        crossover=SimulatedBinaryCrossover(type=2),
    ):
        self.lb = lb
        self.ub = ub
        self.n_objs = n_objs
        self.dim = lb.shape[0]
        self.k = k
        self.pop_size = (jnp.ceil(pop_size / self.k) * self.k).astype(int)
        # self.type = type
        # self.T = jnp.ceil(self.pop_size / 10).astype(int)

        self.mutation = mutation
        self.crossover = crossover

    def setup(self, key):
        key, subkey1, subkey2 = jax.random.split(key, 3)
        population = (
            jax.random.uniform(subkey1, shape=(self.pop_size, self.dim))
            * (self.ub - self.lb)
            + self.lb
        )
        # w = UniformSampling(self.pop_size, self.n_objs).random()[0]
        w = LatinHypercubeSampling(self.k, self.n_objs).random(subkey2)[0]
        s = (self.pop_size / self.k).astype(int)
        # B = euclidean_dis(w, w)
        # B = jnp.argsort(B, axis=1)
        # B = B[:, : self.T]
        return State(
            population=population,
            fitness=jnp.zeros((self.pop_size, self.n_objs)),
            next_generation=population,
            weight_vector=w,
            # B=B,
            Z=jnp.zeros(shape=self.n_objs),
            parent=jnp.zeros((self.pop_size, self.T)).astype(int),
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
        key, subkey = jax.random.split(state.key)
        parent = jax.random.permutation(
            subkey, state.B, axis=1, independent=True
        ).astype(int)
        population = state.population
        selected_p = jnp.r_[population[parent[:, 0]], population[parent[:, 1]]]

        crossovered, state = self.crossover(state, selected_p)
        next_generation, state = self.mutation(state, crossovered, (self.lb, self.ub))
        # next_generation = jnp.clip(mutated, self.lb, self.ub)

        return next_generation, state.update(
            next_generation=next_generation, parent=parent, key=key
        )

    def _tell_init(self, state, fitness):
        Z = jnp.min(fitness, axis=0)
        state = state.update(fitness=fitness, Z=Z, is_init=False)
        return state

    def _tell_normal(self, state, fitness):
        population = state.population
        pop_obj = state.fitness
        offspring = state.next_generation
        obj = fitness
        w = state.weight_vector
        Z = state.Z
        parent = state.parent

        out_vals = (population, pop_obj, Z)

        def out_body(i, out_vals):
            population, pop_obj, Z = out_vals
            ind_p = parent[i]
            ind_obj = obj[i]
            Z = jnp.minimum(Z, obj[i])

            if self.type == 1:
                # PBI approach
                norm_w = jnp.linalg.norm(w[ind_p], axis=1)
                norm_p = jnp.linalg.norm(
                    pop_obj[ind_p] - jnp.tile(Z, (self.T, 1)), axis=1
                )
                norm_o = jnp.linalg.norm(ind_obj - Z)
                cos_p = (
                    jnp.sum(
                        (pop_obj[ind_p] - jnp.tile(Z, (self.T, 1))) * w[ind_p], axis=1
                    )
                    / norm_w
                    / norm_p
                )
                cos_o = (
                    jnp.sum(jnp.tile(ind_obj - Z, (self.T, 1)) * w[ind_p], axis=1)
                    / norm_w
                    / norm_o
                )
                g_old = norm_p * cos_p + 5 * norm_p * jnp.sqrt(1 - cos_p**2)
                g_new = norm_o * cos_o + 5 * norm_o * jnp.sqrt(1 - cos_o**2)
            if self.type == 2:
                # Tchebycheff approach
                g_old = jnp.max(
                    jnp.abs(pop_obj[ind_p] - jnp.tile(Z, (self.T, 1))) * w[ind_p],
                    axis=1,
                )
                g_new = jnp.max(
                    jnp.tile(jnp.abs(ind_obj - Z), (self.T, 1)) * w[ind_p], axis=1
                )
            if self.type == 3:
                # Tchebycheff approach with normalization
                z_max = jnp.max(pop_obj, axis=0)
                g_old = jnp.max(
                    jnp.abs(pop_obj[ind_p] - jnp.tile(Z, (self.T, 1)))
                    / jnp.tile(z_max - Z, (self.T, 1))
                    * w[ind_p],
                    axis=1,
                )
                g_new = jnp.max(
                    jnp.tile(jnp.abs(ind_obj - Z), (self.T, 1))
                    / jnp.tile(z_max - Z, (self.T, 1))
                    * w[ind_p],
                    axis=1,
                )
            if self.type == 4:
                # Modified Tchebycheff approach
                g_old = jnp.max(
                    jnp.abs(pop_obj[ind_p] - jnp.tile(Z, (self.T, 1))) / w[ind_p],
                    axis=1,
                )
                g_new = jnp.max(
                    jnp.tile(jnp.abs(ind_obj - Z), (self.T, 1)) / w[ind_p], axis=1
                )

            g_new = g_new[:, jnp.newaxis]
            g_old = g_old[:, jnp.newaxis]
            population = population.at[ind_p].set(
                jnp.where(g_old >= g_new, offspring[ind_p], population[ind_p])
            )
            pop_obj = pop_obj.at[ind_p].set(
                jnp.where(g_old >= g_new, obj[ind_p], pop_obj[ind_p])
            )

            return (population, pop_obj, Z)

        population, pop_obj, Z = jax.lax.fori_loop(0, self.pop_size, out_body, out_vals)

        state = state.update(population=population, fitness=pop_obj, Z=Z)
        return state
