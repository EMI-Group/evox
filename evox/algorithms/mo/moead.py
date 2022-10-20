import evox as ex
import jax
import jax.numpy as jnp
import jax.experimental.host_callback as hcb
from evox.operators.mutation import PmMutation
from evox.operators.crossover import SimulatedBinaryCrossover
from evox.operators import uniform_point
from evox.utils import euclidean_dis


@ex.jit_class
class MOEAD(ex.Algorithm):
    """MOEA/D algorithm

    link: https://ieeexplore.ieee.org/document/4358754
    """

    def __init__(
        self,
        lb,
        ub,
        n_objs,
        pop_size,
        type=1,
        mutation=PmMutation(),
        crossover=SimulatedBinaryCrossover(type=2),
    ):
        self.lb = lb
        self.ub = ub
        self.n_objs = n_objs
        self.dim = lb.shape[0]
        self.pop_size = pop_size
        self.type = type
        self.T = jnp.ceil(self.pop_size/10).astype(int)

        self.mutation = mutation
        self.crossover = crossover

    def setup(self, key):
        key, subkey = jax.random.split(key)
        population = (
            jax.random.uniform(subkey, shape=(self.pop_size, self.dim))
            * (self.ub - self.lb)
            + self.lb
        )
        w = uniform_point(self.pop_size, self.n_objs)[0]
        B = euclidean_dis(w, w)
        B = jnp.argsort(B, axis=1)
        B = B[:, :self.T]
        return ex.State(
            population=population,
            fitness=jnp.zeros((self.pop_size, self.n_objs)),
            next_generation=population,
            weight_vector=w,
            B=B,
            Z=jnp.zeros(shape=self.n_objs),
            parent=jnp.zeros((self.pop_size, self.T)).astype(int),
            is_init=True,
            key=key,)

    @ex.jit_method
    def ask(self, state):
        return jax.lax.cond(state.is_init, self._ask_init, self._ask_normal, state)

    def tell(self, state, fitness):
        return jax.lax.cond(state.is_init,
                            self._tell_init,
                            self._tell_normal,
                            state, fitness
                            )

    def _ask_init(self, state):
        return state, state.population

    def _ask_normal(self, state):
        key, subkey = jax.random.split(state.key)
        parent = jax.random.permutation(
            subkey, state.B, axis=1, independent=True).astype(int)
        population = state.population
        selected_p = jnp.r_[population[parent[:, 0]], population[parent[:, 1]]]

        state, crossovered = self.crossover(state, selected_p)
        state, mutated = self.mutation(state, crossovered, (self.lb, self.ub))
        next_generation = jnp.clip(mutated, self.lb, self.ub)

        return state.update(next_generation=next_generation, parent=parent, key=key), next_generation

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
        for i in range(self.pop_size):
            ind_p = parent[i]
            ind_obj = obj[i]
            Z = jnp.minimum(Z, obj[i])

            if self.type == 1:
                # PBI approach
                norm_w = jnp.linalg.norm(w[ind_p], axis=1)
                norm_p = jnp.linalg.norm(
                    pop_obj[ind_p] - jnp.tile(Z, (self.T, 1)), axis=1)
                norm_o = jnp.linalg.norm(ind_obj - Z)
                cos_p = jnp.sum(
                    (pop_obj[ind_p] - jnp.tile(Z, (self.T, 1))) * w[ind_p], axis=1) / norm_w / norm_p
                cos_o = jnp.sum(jnp.tile(ind_obj - Z, (self.T, 1))
                                * w[ind_p], axis=1) / norm_w / norm_o
                g_old = norm_p * cos_p + 5 * norm_p * jnp.sqrt(1 - cos_p ** 2)
                g_new = norm_o * cos_o + 5 * norm_o * jnp.sqrt(1 - cos_o ** 2)
            if self.type == 2:
                # Tchebycheff approach
                g_old = jnp.max(
                    jnp.abs(pop_obj[ind_p] - jnp.tile(Z, (self.T, 1))) * w[ind_p], axis=1)
                g_new = jnp.max(jnp.tile(jnp.abs(ind_obj - Z),
                                (self.T, 1)) * w[ind_p], axis=1)
            if self.type == 3:
                # Tchebycheff approach with normalization
                z_max = jnp.max(pop_obj, axis=0)
                g_old = jnp.max(jnp.abs(pop_obj[ind_p] - jnp.tile(Z, (self.T, 1))) / jnp.tile(
                    z_max - Z, (self.T, 1)) * w[ind_p], axis=1)
                g_new = jnp.max(jnp.tile(jnp.abs(ind_obj - Z), (self.T, 1)) /
                                jnp.tile(z_max - Z, (self.T, 1)) * w[ind_p], axis=1)
            if self.type == 4:
                # Modified Tchebycheff approach
                g_old = jnp.max(
                    jnp.abs(pop_obj[ind_p] - jnp.tile(Z, (self.T, 1))) / w[ind_p], axis=1)
                g_new = jnp.max(jnp.tile(jnp.abs(ind_obj - Z),
                                (self.T, 1)) / w[ind_p], axis=1)

            temp_pop = population
            temp_obj = pop_obj
            for j in range(self.T):
                population = jax.lax.cond(g_old[j] >= g_new[j], lambda: population.at[ind_p[j]].set(
                    offspring[j]), lambda: population.at[ind_p[j]].set(temp_pop[ind_p[j]]))
                pop_obj = jax.lax.cond(g_old[j] >= g_new[j], lambda: pop_obj.at[ind_p[j]].set(
                    obj[j]), lambda: pop_obj.at[ind_p[j]].set(temp_obj[ind_p[j]]))

        state = state.update(population=population, fitness=pop_obj, Z=Z)
        return state
