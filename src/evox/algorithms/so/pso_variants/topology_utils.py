import jax
import jax.numpy as jnp
from .utils import (
    row_argsort,
    get_distance_matrix,
    select_from_mask,
)
from functools import partial
from typing import Union, Iterable, Literal
import math
import numpy


@partial(jax.jit, static_argnums=[2, 3])
def get_circles_neighbour(
    random_key: jax.Array, population: jax.Array, K: int, shortcut: int
):
    """
    each individual is connected to its K immediate neighbors only.
    K=2 by default.
    shortcut is the numebr of edges changed after connection.
    returns a N*N ,each row indicates the indices of neighbours, if not enough neighbour, pending it's own indice.
    """
    dist_matrix = get_distance_matrix(population)

    row_dist_rank = row_argsort(dist_matrix)

    adjacancy_matrix = build_adjacancy_matrix_by_K_nearest_neighbour(row_dist_rank, K)

    # jax.debug.print("adjacancy_matrix: {}", adjacancy_matrix)
    adjacancy_matrix = mutate_shortcut(
        key=random_key, adjacancy_matrix=adjacancy_matrix, num_shortcut=shortcut
    )

    # jax.debug.print("adjacancy_matrix_mutated: {}", adjacancy_matrix)

    return adjacancy_matrix


@jax.jit
def get_full_neighbour(population: jax.Array):
    """
    fully connected
    """
    N = population.shape[0]
    adjacancy_matrix = jnp.ones((N, N), dtype="int32")
    return adjacancy_matrix


@partial(jax.jit, static_argnums=[1])
def get_ring_neighbour(population: jax.Array, K: int):
    """
    every individual is connected to two others;
    No more details, we apply same implementation from PyGMO:
    ''The L-best-k topology consists of n nodes arranged in a ring, in which
    node i is connected to each node in {(i+j) mod n : j = +-1,+-2, . . . ,+-k}.'' \n
    [Mohais et al., 2005] http://dx.doi.org/10.1007/11589990_80
    """
    N = population.shape[0]
    adjacancy_matrix = jnp.triu(jnp.ones(shape=(N, N)), k=-K)
    adjacancy_matrix = jnp.tril(adjacancy_matrix, k=K)
    adjacancy_matrix = (
        adjacancy_matrix
        + jnp.triu(jnp.ones(shape=(N, N)), k=N - K)
        + jnp.tril(jnp.ones(shape=(N, N)), k=K - N)
    )
    return adjacancy_matrix


@jax.jit
def get_square_neighbour(population: jax.Array):
    """
    "The square is a graph representing a rectangular lattice that folds like a torus.
    This structure, albeit artificial, is commonly used to represent neighborhoods in the Evolutionary Computation and Cellular Automata communities,
    and is referred to as the von Neumann neighborhood"
    according to "The Fully Informed Particle Swarm: Simpler, Maybe Better",2004

    according to PyGMO, reshape N individual into row*col where number rows and cols are as close as possible.
    Each connect to the upper, lower, left and right neighbour

    """
    N = population.shape[0]
    col = math.floor(math.sqrt(N))
    while col > 1 and N % col != 0:
        col -= 1
    row = N // col
    # NOTE: this will result in a ring topology when col==1
    if col <= 2:
        # TODO: warn the users if possible
        print(
            f"Population size is {N}, When creating square topology, number of rows cols {row}*{col} may cause strange topo"
        )

    np_mat = numpy.arange(0, N).reshape((col, row))
    dir = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    print(np_mat)
    adj_mat = numpy.zeros((N, N))
    for i in range(col):
        for j in range(row):
            x = np_mat[i, j]
            for k in range(4):
                y = np_mat[(i + col + dir[k][0]) % col, (i + row + dir[k][1]) % row]
                adj_mat[x, y] = 1
    print(adj_mat)
    adj_mat = adj_mat.tolist()
    adjacancy_matrix = jnp.array(adj_mat)
    return adjacancy_matrix


@jax.jit
def get_neighbour_best_fitness(fitness, adjacancy_list):
    """
    given an adjacancy list, and its masking output the neighbour fitness
    """

    @jax.jit
    def argmin_by_row(index: jax.Array):
        argmin_val = jnp.argmin(fitness[index])
        # jax.debug.print("argminval {}", argmin_val)
        return index[argmin_val]

    neighbour_best_indice = jax.vmap(argmin_by_row, in_axes=0)(adjacancy_list)
    neighbour_best_fitness = fitness[neighbour_best_indice]
    return neighbour_best_fitness, neighbour_best_indice


@partial(jax.jit, static_argnums=[1])
def build_adjacancy_matrix_by_K_nearest_neighbour(distance_ranking, K):
    """
    build adjacancy matrix by k nearest neighbour.
    input N*N distance_ranking.
    get N*(K+1) adjacancy list (duplicate).
    flatten to get indices.
    form a new adjacancy matrix.
    """
    N = distance_ranking.shape[0]

    row_idx = jnp.arange(N)
    row_idx = jnp.broadcast_to(row_idx[:, jnp.newaxis], (N, K + 1))
    index = jnp.take(distance_ranking, indices=jnp.arange(K + 1), axis=1)
    index = index.flatten()
    row_idx = row_idx.flatten()

    index_X = jnp.hstack([index, row_idx])
    index_Y = jnp.hstack([row_idx, index])

    A = jnp.zeros((N, N), dtype=index_X.dtype)
    A = jax.lax.scatter(
        A,
        jnp.vstack([index_X, index_Y]).T,
        jnp.ones_like(index_X),
        jax.lax.ScatterDimensionNumbers((), (0, 1), (0, 1)),
        mode="promise_in_bounds",
    )

    return A


@partial(jax.jit, static_argnums=[1])
def build_adjacancy_list_from_matrix(adjacancy_matrix, keep_self_loop=True):
    """
    given N*N adjacancy matrix, for every row i, output the outgoing neibour in length N.
    fill the rest as its own indice.
    output the masking at the sametime, indicating the mapping.
    """
    N = adjacancy_matrix.shape[0]

    adjacancy_list_masking = jnp.ones((N, N), dtype=adjacancy_matrix.dtype)

    def get_row_indices(indices):
        (row_indices,) = jax.numpy.nonzero(
            indices, size=N, fill_value=-1
        )  # fill value < 0,
        return row_indices

    adjacancy_list = jax.vmap(get_row_indices, in_axes=0)(adjacancy_matrix)

    adjacancy_list_masking = jnp.where(adjacancy_list == -1, 0, adjacancy_list_masking)

    row_idxs = jnp.arange(N, dtype=adjacancy_matrix.dtype)
    identity = jnp.stack([row_idxs for _ in range(N)], axis=1)

    if not keep_self_loop:
        adjacancy_list_masking = jnp.where(
            adjacancy_list == identity, 0, adjacancy_list_masking
        )

    adjacancy_list = jnp.where(adjacancy_list == -1, identity, adjacancy_list).astype(
        "int32"
    )

    return adjacancy_list, adjacancy_list_masking


@partial(jax.jit, static_argnums=[2])
def mutate_shortcut(key: jax.Array, adjacancy_matrix: jax.Array, num_shortcut: int):
    """
    given an N*N adjacancy matrix. (indicating undirected connections)
    change num_shortcut pairs of edge to others.
    """
    N = adjacancy_matrix.shape[0]
    k1, k2, k3, k4 = jax.random.split(key=key, num=4)

    diag_matrix = jnp.ones(shape=(N,), dtype=adjacancy_matrix.dtype)
    diag_matrix = jnp.diag(diag_matrix, k=0)

    flatten = jnp.triu(
        adjacancy_matrix - diag_matrix
    ).flatten()  # get upper triangle matrix, remvoe duplicate edges
    rows = jnp.repeat(jnp.arange(N), N)
    cols = jnp.tile(jnp.arange(N), N)

    # the mask of whether the connection is going to mutate
    mutate_mask = select_from_mask(k1, flatten, num_shortcut)

    # mutate all connections (may have duplicate edges)
    mutate_row = jax.random.choice(k2, jnp.array([True, False]), (N * N,))
    mutate_col = ~mutate_row
    mutate_nodes = jax.random.choice(k3, jnp.arange(N), (N * N,))
    new_rows = jnp.where(mutate_row, mutate_nodes, rows)
    new_cols = jnp.where(mutate_col, mutate_nodes, cols)

    # mutate all connections (try to ignore duplicate edges but too hard...)

    # prob_matrix = jnp.triu(prob_matrix) # get upper triangle matrix, remvoe duplicate edges
    # prob_matrix_row = jax.random.uniform(k2, shape=(N,N), minval=3, maxval=4)
    # prob_matrix_row  = prob_matrix_row - adjacancy_matrix - 2*diag_matrix
    # prob_matrix_col = jax.random.uniform(k3, shape=(N,N), minval=3, maxval=4)
    # prob_matrix_col  = prob_matrix_col - adjacancy_matrix - 2*diag_matrix

    # mutate_row = jax.random.choice(k2, jnp.array([True, False]), (N * N, ))
    # mutate_col = ~mutate_row

    # candidate_rows = ....
    # candidate_cols = ....
    # new_rows = jnp.where(mutate_row, candidate_rows, rows)
    # new_cols = jnp.where(mutate_col, candidate_cols, cols)

    # only mutate the selected connections
    rows = jnp.where(mutate_mask, new_rows, rows)
    cols = jnp.where(mutate_mask, new_cols, cols)

    # unflatten
    mutated_adjacancy_matrix = (
        jnp.zeros_like(adjacancy_matrix).at[rows, cols].add(flatten)
    )
    mutated_adjacancy_matrix = jnp.clip(
        mutated_adjacancy_matrix + mutated_adjacancy_matrix.T + diag_matrix, 0, 1
    )

    return mutated_adjacancy_matrix
