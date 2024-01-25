# Q. Zhang, A. Zhou, and Y. Jin, RM-MEDA: A regularity model-based multiobjective estimation of distribution algorithm,
# IEEE Transactions on Evolutionary Computation, 2008, 12(1): 41-63.


import jax
import jax.numpy as jnp
from jax import jit, vmap, random
from evox import Algorithm, jit_class, State
from evox.operators import (
    non_dominated_sort,
    crowding_distance,
    selection,
    mutation,
)
import jax.lax as lax
from functools import partial

# @partial(jax.jit, static_argnums=[0, 1, 2])
def local_pca(pop_dec, M, K, key):
    n, d = pop_dec.shape  # Dimensions
    # d = 12

    # model = [
    #     {
    #         "mean": pop_dec[k],  # The mean of the model
    #         "PI": jnp.eye(d),  # The matrix PI
    #         "e_vector": [],  # The eigenvectors
    #         "e_value": [],  # The eigenvalues
    #         "a": [],  # The lower bound of the projections
    #         "b": [],
    #     }
    #     for k in range(K)
    # ]  # The upper bound of the projections

    def create_model(k):
        return {
            "mean": pop_dec[k],  # The mean of the model
            "PI": jnp.eye(d),  # The matrix PI
            "e_vector": jnp.array([]),  # The eigenvectors
            "e_value": jnp.array([]),  # The eigenvalues
            "a": jnp.array([]),  # The lower bound of the projections
            "b": jnp.array([]),  # The upper bound of the projections
        }
    arr = jnp.arange(K)
    model = jax.vmap(create_model)(arr)

    ## Modeling
    for iteration in range(50):
        # Calculte the distance between each solution and its projection in affine principal subspace of each cluster
        def distance_fun(k):
            diff = pop_dec - jnp.tile(model["mean"][k], (n, 1))
            return jnp.sum(diff.dot(model["PI"][k]) * diff, axis=1)
        distance = jax.vmap(distance_fun, out_axes=1)(arr)
        partition = jnp.argmin(distance, axis=1)
        # Update the model of each cluster
        updated = jnp.zeros(K, dtype=bool)
        for k in range(K):
        # def body_fun2(k):
            old_mean = model["mean"][k]
            # select current cluster
            current = partition == k
            if jnp.sum(current) < 2:
                if not jnp.any(current):
                    current = random.randint(shape=(1,), key=key, maxval=n, minval=0)
                # def false_fun2(current):
                #     random_index = random.randint(shape=(1,), key=key, maxval=n, minval=0)
                #     bool_matrix = jnp.equal(jnp.arange(current.size), random_index)
                #     # jnp.array([random.randint(shape=(1,), key=key, maxval=n, minval=0)], dtype=int)
                #     return bool_matrix
                # current = lax.cond(jnp.any(current), lambda x: x, false_fun2, current)
                model["mean"] = model['mean'].at[k].set(pop_dec[current].reshape(-1))
                model["PI"] = model["PI"].at[k].set(jnp.eye(d))
                model["e_vector"] = model["e_vector"].at[k].set([])
                model["e_value"] = model["e_value"].at[k].set([])
            else:
                model["mean"] = model["mean"].at[k].set(jnp.mean(pop_dec[current], axis=0))
                cc = jnp.cov(
                    (
                        pop_dec[current]
                        - jnp.tile(model["mean"][k], (jnp.sum(current), 1))
                    ).T
                )
                # Using eigh for Hermitian matrices
                e_value, e_vector = jnp.linalg.eigh(
                    cc
                )
                rank = jnp.argsort(e_value)
                e_value = jnp.sort(e_value)
                model["e_value"] = model["e_value"].at[k].set(e_value)
                model["e_vector"] = model["e_vector"].at[k].set(e_vector[:, rank])
                # Note: this code using maximum eigenvalues instead of minimum eigenvalues in PlateEmo
                model["PI"] = model["PI"].at[k].set(model["e_vector"][k:k+1, (M - 1):].dot(
                    model["e_vector"][k:k+1, (M - 1):].conj().transpose()
                ))
            updated = updated.at[k].set(
                (not jnp.any(current))
                or (jnp.sqrt(jnp.sum((old_mean - model["mean"][k]) ** 2)) > 1e-5)
            )
        # Break if no change is made
        if not jnp.any(updated):
            break

    # Calculate the smallest hyper-rectangle of each model
    # for k in range(K):
    def body_fun(k):
        if len(model["e_vector"][k]) != 0:
            hyper_rectangle = (
                pop_dec[partition == k]
                - jnp.tile(model["mean"][k], (jnp.sum(partition == k), 1))
            ).dot(model["e_vector"][k][:, : M - 1])
            model["a"] = model['a'].at[k].set(jnp.min(hyper_rectangle, axis=0))
            model["b"] = model['b'].at[k].set(jnp.max(hyper_rectangle, axis=0))  # this should by tested
        else:
            model["a"] = model["a"].at[k].set(jnp.zeros(M - 1))
            model["b"] = model["b"].at[k].set(jnp.zeros(M - 1))
    jax.vmap(body_fun)(arr)
    ## Calculate the probability of each cluster for reproduction
    # Calculate the volume of each cluster
    volume = jnp.asarray([model["b"][k] for k in range(K)]) - jnp.asarray(
        [model["a"][k] for k in range(K)]
    )
    volume = jnp.prod(
        volume, axis=1
    )  # Compute product along axis 1 (rows) to get volumes for each cluster
    # Calculate the cumulative probability of each cluster
    probability = jnp.cumsum(volume / jnp.sum(volume))
    return model, probability


class RMMEDA(Algorithm):
    def __init__(
        self,
        lb,
        ub,
        n_objs=3,
        pop_size=100,
        selection_op=None,
        k_clusters=5,
        mutation_op=None,
    ):
        self.lb = lb
        self.ub = ub
        self.n_objs = n_objs
        self.dim = lb.shape[0]
        self.pop_size = pop_size
        self.k_clusters = k_clusters
        self.selection = selection_op
        self.mutation = mutation_op

        if self.selection is None:
            self.selection = selection.UniformRand(1)
        if self.mutation is None:
            self.mutation = mutation.Polynomial((self.lb, self.ub))

    # initial set
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
            next_generation=population,
            key=key,
        )

    # generate initial population
    def init_ask(self, state):
        return state.population, state

    def init_tell(self, state, fitness):
        state = state.update(fitness=fitness)
        return state

    # generate next generation by mutation
    def ask(self, state):
        population = state.population
        M = self.n_objs
        D = self.dim
        Model, probability = local_pca(
            population, self.n_objs, self.k_clusters, state.key
        )

        OffspringDec = jnp.zeros((self.pop_size, self.dim))
        keys = random.split(state.key, self.pop_size)
        for i in range(self.pop_size):
            k = jnp.argmax(
                probability >= random.uniform(state.key, shape=(self.k_clusters,)),
                axis=0,
            )
            # Generate one offspring
            if len(Model["e_vector"][k]) > 0:  # Check if eVector is not empty
                lower = Model["a"][k] - 0.25 * (Model["b"][k] - Model["a"][k])
                upper = Model["b"][k] + 0.25 * (Model["b"][k] - Model["a"][k])
                trial = (
                    random.uniform(keys[i], shape=(M - 1,)) * (upper - lower) + lower
                )
                sigma = jnp.sum(jnp.abs(Model["e_value"][k][M - 1 : D])) / (D - M + 1)
                OffspringDec = OffspringDec.at[i, :].set(
                    Model["mean"][k]
                    + trial @ Model["e_vector"][k][:, : M - 1].T
                    + (random.normal(keys[i], shape=(D,)) * jnp.sqrt(sigma))
                )
            else:
                OffspringDec = OffspringDec.at[i, :].set(
                    Model["mean"][k] + random.normal(keys[i], shape=(D,))
                )

        return OffspringDec, state.update(next_generation=OffspringDec)

    # select next generation
    def tell(self, state, fitness):
        merged_pop = jnp.concatenate([state.population, state.next_generation], axis=0)
        merged_fitness = jnp.concatenate([state.fitness, fitness], axis=0)
        rank = non_dominated_sort(merged_fitness)
        order = jnp.argsort(rank)
        worst_rank = rank[order[self.pop_size]]
        mask = rank == worst_rank
        last = jnp.sum(mask)
        next = self.pop_size - last
        while last > self.pop_size - next:
            crowding_dis = crowding_distance(merged_fitness, mask)
            order = jnp.argsort(crowding_dis)
            mask[order[-1]] = False
            last -= 1
        survivor = merged_pop[mask]
        survivor_fitness = merged_fitness[mask]
        state = state.update(population=survivor, fitness=survivor_fitness)
        return state
