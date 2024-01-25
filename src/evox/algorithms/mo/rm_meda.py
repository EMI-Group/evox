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


def local_pca(pop_dec, M, K, key):
    n, d = pop_dec.shape  # n: number of solutions, d: number of decision variables

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
    partition = None
    ## Modeling
    def modeling_fun(temp_index, temp):
    # for iteration in range(50):
        # Calculte the distance between each solution and its projection in affine principal subspace of each cluster
        def distance_fun(k):
            diff = pop_dec - jnp.tile(model["mean"][k], (n, 1))
            return jnp.sum(diff.dot(model["PI"][k]) * diff, axis=1)

        distance = jax.vmap(distance_fun, out_axes=1)(arr)
        partition = jnp.argmin(distance, axis=1)
        # In platemo, there is a vector "updated" showing the clustering process. If the change is small, the loop will end. Here is no stop beacuse of JIT
        # updated = jnp.zeros(K, dtype=bool)
        def clustering(k):
        # for k in range(K):
            # old_mean = model["mean"][k]
            # select current cluster
            current = partition == k
            def small_fun(current, model):
                def false_fun(current):
                    current = current.at[random.randint(shape=(1,), key=key, maxval=n, minval=0)].set(True)
                    return current
                current = lax.cond(jnp.any(current), lambda x : x, false_fun, current)
                index = jnp.argmax(current)
                model["mean"] = model["mean"].at[k].set(pop_dec[index].reshape(-1))
                model["PI"] = model["PI"].at[k].set(jnp.eye(d))
                model["e_vector"] = model["e_vector"].at[k].set([])
                model["e_value"] = model["e_value"].at[k].set([])
            def normal_fun(current, model):
            # else:
                cur_pop = jnp.where(current[:, jnp.newaxis], pop_dec, 0)
                model["mean"] = model["mean"].at[k].set(jnp.mean(cur_pop, axis=0))
                c_cov = cur_pop - jnp.tile(model["mean"][k], (len(current), 1))
                c_cov = jnp.where(current[:, jnp.newaxis], c_cov, 0).T
                cc = jnp.cov(c_cov)
                # Using eigh for Hermitian matrices
                e_value, e_vector = jnp.linalg.eigh(cc)
                rank = jnp.argsort(e_value)
                e_value = jnp.sort(e_value)
                _e_vector = e_vector[:, rank]
                model["e_value"] = model["e_value"].at[k].set(e_value)
                model["e_vector"] = model["e_vector"].at[k].set(_e_vector)
                # Note: this code using all eigenvalues instead of maximum eigenvalues because of JIT
                model["PI"] = (
                    model["PI"]
                    .at[k]
                    .set(
                        _e_vector.dot(_e_vector.conj().transpose())
                    )
                )
            lax.cond(jnp.sum(current) < 2, small_fun, normal_fun, current, model)
            # updated = updated.at[k].set(
            #     (jnp.logical_not(jnp.any(current))) | (jnp.sqrt(jnp.sum((old_mean - model["mean"][k]) ** 2)) > 1e-5)
            # )
        jax.vmap(clustering)(jnp.arange(K))
        # Break if no change is made
        # if jnp.logical_not(jnp.any(updated)):
        #     break

    lax.fori_loop(0,50, modeling_fun, None)
    # Calculate the smallest hyper-rectangle of each model
    def body_fun(k):
        if len(model["e_vector"][k]) != 0:
            hyper_rectangle = (
                pop_dec[partition == k]
                - jnp.tile(model["mean"][k], (jnp.sum(partition == k), 1))
            ).dot(model["e_vector"][k][:, : M - 1])
            model["a"] = model["a"].at[k].set(jnp.min(hyper_rectangle, axis=0))
            model["b"] = (
                model["b"].at[k].set(jnp.max(hyper_rectangle, axis=0))
            )  # this should by tested
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
        crowding_dis = crowding_distance(merged_fitness, mask)

        combined_order = jnp.lexsort((-crowding_dis, rank))[: self.pop_size]
        survivor = merged_pop[combined_order]
        survivor_fitness = merged_fitness[combined_order]
        state = state.update(population=survivor, fitness=survivor_fitness)
        return state
