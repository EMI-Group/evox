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


def local_pca(pop_dec, M, K, key):
    n, d = pop_dec.shape  # Dimensions
    model = [
        {
            "mean": pop_dec[k],  # The mean of the model
            "PI": jnp.eye(d),  # The matrix PI
            "e_vector": [],  # The eigenvectors
            "e_value": [],  # The eigenvalues
            "a": [],  # The lower bound of the projections
            "b": [],
        }
        for k in range(K)
    ]  # The upper bound of the projections

    ## Modeling
    for iteration in range(50):
        # Calculte the distance between each solution and its projection in affine principal subspace of each cluster
        distance = jnp.zeros((n, K))  # matrix of zeros N*K
        for k in range(K):
            diff = pop_dec - jnp.tile(model[k]["mean"], (n, 1))
            distance = distance.at[:, k].set(
                jnp.sum(diff.dot(model[k]["PI"]) * diff, axis=1)
            )
        # Partition
        partition = jnp.argmin(distance, axis=1)
        # Update the model of each cluster
        updated = jnp.zeros(K, dtype=bool)
        for k in range(K):
            old_mean = model[k]["mean"]
            # select current cluster
            current = partition == k
            if jnp.sum(current) < 2:
                if not jnp.any(current):
                    current = jnp.array(
                        [random.randint(shape=(1,), key=key, maxval=n, minval=0)],
                        dtype=int,
                    )
                model[k]["mean"] = pop_dec[current]
                model[k]["PI"] = jnp.eye(d)
                model[k]["e_vector"] = []
                model[k]["e_value"] = []
            else:
                model[k]["mean"] = jnp.mean(pop_dec[current], axis=0)
                cc = jnp.cov(
                    (
                        pop_dec[current]
                        - jnp.tile(model[k]["mean"], (jnp.sum(current), 1))
                    ).T
                )
                e_value, e_vector = jnp.linalg.eigh(
                    cc
                )  # Using eigh for Hermitian matrices
                rank = jnp.argsort(e_value)
                e_value = jnp.sort(e_value)
                model[k]["e_value"] = e_value
                model[k]["e_vector"] = e_vector[:, rank]
                # Note: this code using maximum eigenvalues instead of minimum eigenvalues in PlateEmo
                model[k]["PI"] = model[k]["e_vector"][:, (M - 1) :].dot(
                    model[k]["e_vector"][:, (M - 1) :].T
                )

            updated = updated.at[k].set(
                (not jnp.any(current))
                or (jnp.sqrt(jnp.sum((old_mean - model[k]["mean"]) ** 2)) > 1e-5)
            )

        # Break if no change is made
        if not jnp.any(updated):
            break
    # Calculate the smallest hyper-rectangle of each model
    for k in range(K):
        if len(model[k]["e_vector"]) != 0:
            hyper_rectangle = (
                pop_dec[partition == k]
                - jnp.tile(model[k]["mean"], (jnp.sum(partition == k), 1))
            ).dot(model[k]["e_vector"][:, : M - 1])
            model[k]["a"] = jnp.min(hyper_rectangle, axis=0)
            model[k]["b"] = jnp.max(hyper_rectangle, axis=0)  # this should by tested
        else:
            model[k]["a"] = jnp.zeros(M - 1)
            model[k]["b"] = jnp.zeros(M - 1)

    ## Calculate the probability of each cluster for reproduction
    # Calculate the volume of each cluster
    volume = jnp.asarray([model[k]["b"] for k in range(K)]) - jnp.asarray(
        [model[k]["a"] for k in range(K)]
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
            if len(Model[k]["e_vector"]) > 0:  # Check if eVector is not empty
                lower = Model[k]["a"] - 0.25 * (Model[k]["b"] - Model[k]["a"])
                upper = Model[k]["b"] + 0.25 * (Model[k]["b"] - Model[k]["a"])
                trial = (
                    random.uniform(keys[i], shape=(M - 1,)) * (upper - lower) + lower
                )
                sigma = jnp.sum(jnp.abs(Model[k]["e_value"][M - 1 : D])) / (D - M + 1)
                OffspringDec = OffspringDec.at[i, :].set(
                    Model[k]["mean"]
                    + trial @ Model[k]["e_vector"][:, : M - 1].T
                    + (random.normal(keys[i], shape=(D,)) * jnp.sqrt(sigma))
                )
            else:
                OffspringDec = OffspringDec.at[i, :].set(
                    Model[k]["mean"] + random.normal(keys[i], shape=(D,))
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
