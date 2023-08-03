import jax
import jax.numpy as jnp

<<<<<<< HEAD
from evox import jit_class, Algorithm, State, Operator
from evox.operators.selection import UniformRandomSelection
from evox.operators.mutation import PmMutation
from evox.operators.crossover import SimulatedBinaryCrossover
from evox.operators.sampling import UniformSampling, LatinHypercubeSampling
from evox.utils import cos_dist


@jit_class
class ReferenceVectorGuidedSelection(Operator):
    """Reference vector guided environmental selection.

    """

    def __init__(self, x=None, v=None, theta=None):
        self.x = x
        self.v = v
        self.theta = theta

    def setup(self, key):
        return State(key=key)

    def __call__(self, state, x, v, theta):
        self.x = x
        self.v = v
        self.theta = theta
        key, subkey = jax.random.split(state.key)
        n, m = jnp.shape(self.x)
        nv = jnp.shape(self.v)[0]
        obj = self.x

        obj -= jnp.tile(jnp.min(obj, axis=0), (n, 1))

        cosine = cos_dist(self.v, self.v)
        cosine = jnp.where(jnp.eye(jnp.shape(cosine)[0], dtype=bool), 0, cosine)
        cosine = jnp.clip(cosine, -1, 1)
        gamma = jnp.min(jnp.arccos(cosine), axis=1)

        angle = jnp.arccos(cos_dist(obj, self.v))

        associate = jnp.argmin(angle, axis=1)

        next_ind = jnp.full(nv, -1)
        is_null = jnp.sum(next_ind)

        global_min = jnp.inf
        global_min_idx = -1

        vals = next_ind, global_min, global_min_idx

        def update_next(i, sub_index, next_ind, global_min, global_min_idx):
            apd = (1+m*theta*angle[sub_index, i]/gamma[i]) * jnp.sqrt(jnp.sum(obj[sub_index, :]**2, axis=1))

            apd_max = jnp.max(apd)
            noise = jnp.where(sub_index == -1, apd_max, 0)
            apd = apd + noise
            best = jnp.argmin(apd)

            global_min_idx = jnp.where(apd[best] < global_min, sub_index[best.astype(int)], global_min_idx)
            global_min = jnp.minimum(apd[best], global_min)

            next_ind = next_ind.at[i].set(sub_index[best.astype(int)])
            return next_ind, global_min, global_min_idx

        def no_update(i, sub_index, next_ind, global_min, global_min_idx):
            return next_ind, global_min, global_min_idx

        def body_fun(i, vals):
            next_ind, global_min, global_min_idx = vals
            sub_index = jnp.where(associate == i, size=nv, fill_value=-1)[0]

            next_ind, global_min, global_min_idx = jax.lax.cond(jnp.sum(sub_index) != is_null, update_next, no_update, i, sub_index, next_ind, global_min, global_min_idx)
            return next_ind, global_min, global_min_idx

        next_ind, global_min, global_min_idx = jax.lax.fori_loop(0, nv, body_fun, vals)
        mask = next_ind == -1

        next_ind = jnp.where(mask, global_min_idx, next_ind)
        next_ind = jnp.where(global_min_idx != -1, next_ind, jnp.arange(0, nv))

        return next_ind, State(key=key)


@jit_class
=======
from evox.operators import selection, mutation, crossover
from evox.operators.sampling import UniformSampling, LatinHypercubeSampling
from evox import Algorithm, State, jit_class


@jit_class
>>>>>>> e9d1a7a9a7ff3bb82fc0c14c4cd4180929c822b9
class RVEA(Algorithm):
    """RVEA algorithms

    link: https://ieeexplore.ieee.org/document/7386636
    """

    def __init__(
        self,
        lb,
        ub,
        n_objs,
        pop_size,
        alpha=2,
        fr=0.1,
        max_gen=100,
<<<<<<< HEAD
        rs=50,
        mating=UniformRandomSelection(p=1),
        selection=ReferenceVectorGuidedSelection(),
        mutation=PmMutation(),
        crossover=SimulatedBinaryCrossover(),
=======
        selection_op=None,
        mutation_op=None,
        crossover_op=None,
>>>>>>> e9d1a7a9a7ff3bb82fc0c14c4cd4180929c822b9
    ):
        self.lb = lb
        self.ub = ub
        self.n_objs = n_objs
        self.dim = lb.shape[0]
        self.pop_size = pop_size
        self.alpha = alpha
        self.fr = fr
        self.max_gen = max_gen
        self.rs = rs

<<<<<<< HEAD
        self.mating = mating
        self.selection = selection
        self.mutation = mutation
        self.crossover = crossover
=======
        self.selection = selection_op
        self.mutation = mutation_op
        self.crossover = crossover_op

        if self.selection is None:
            self.selection = selection.ReferenceVectorGuided()
        if self.mutation is None:
            self.mutation = mutation.Polynomial((lb, ub))
        if self.crossover is None:
            self.crossover = crossover.SimulatedBinary()
>>>>>>> e9d1a7a9a7ff3bb82fc0c14c4cd4180929c822b9
        # self.sampling = UniformSampling(self.pop_size, self.n_objs)
        self.sampling = LatinHypercubeSampling(self.pop_size, self.n_objs)

    def setup(self, key):
        key, subkey1, subkey2 = jax.random.split(key, 3)
        population = (
            jax.random.uniform(subkey1, shape=(self.pop_size, self.dim))
            * (self.ub - self.lb)
            + self.lb
        )
        v = self.sampling.random(subkey2)[0]
        v = v / jnp.linalg.norm(v, axis=0)

        return State(
            population=population,
            fitness=jnp.zeros((self.pop_size, self.n_objs)),
            next_generation=population,
            reference_vector=v,
            is_init=True,
            key=key,
            gen=0
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
<<<<<<< HEAD

        key = state.key
        key, subkey = jax.random.split(key)

        mating_pool = jax.random.permutation(subkey, jnp.arange(0, self.pop_size))
        # mating_pool, state = self.mating(state, state.population)
        population = state.population[mating_pool]
        crossovered, state = self.crossover(state, population)
        next_generation, state = self.mutation(state, crossovered, (self.lb, self.ub))
=======
        mask = state.mask
        key, subkey, x_key, mut_key = jax.random.split(state.key, 4)
        r = (
            jax.random.uniform(subkey, shape=(self.pop_size, self.dim))
            * (self.ub - self.lb)
            + self.lb
        )

        crossovered = self.crossover(x_key, state.population)
        next_generation = self.mutation(mut_key, crossovered)
        next_generation = jnp.where(mask, r, next_generation)
>>>>>>> e9d1a7a9a7ff3bb82fc0c14c4cd4180929c822b9

        return next_generation, state.update(next_generation=next_generation, key=key)

    def _tell_init(self, state, fitness):
        state = state.update(fitness=fitness, is_init=False)
        return state

    def _tell_normal(self, state, fitness):
        current_gen = state.gen + 1
        v = state.reference_vector

        merged_pop = jnp.concatenate([state.population, state.next_generation], axis=0)
        merged_fitness = jnp.concatenate([state.fitness, fitness], axis=0)

        rank = self.selection(
            merged_fitness, v, (current_gen / self.max_gen) ** self.alpha
        )

        survivor = merged_pop[rank]
        survivor_fitness = merged_fitness[rank]

        def rv_adaptation(pop_obj, v):
            v_temp = v * jnp.tile((pop_obj.max(0) - pop_obj.min(0)), (len(v), 1))
            next_v = v.astype(float)

            next_v = v_temp / jnp.tile(
                jnp.sqrt(jnp.sum(v_temp**2, axis=1)).reshape(len(v), 1),
                (1, jnp.shape(v)[1]),
            )

            return next_v

        def no_update(_pop_obj, v):
            return v

        v = jax.lax.cond(
            current_gen % (1 / self.fr) == 0,
            rv_adaptation,
            no_update,
            survivor_fitness,
            v,
        )

        state = state.update(
            population=survivor,
            fitness=survivor_fitness,
            reference_vector=v,
            gen=current_gen,
        )
        return state

