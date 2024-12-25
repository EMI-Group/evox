import torch
import math
import numpy as np

from .utils import (
    row_argsort,
    get_distance_matrix,
    select_from_mask,
)
def get_circles_neighbour( 
    population: torch.Tensor, 
    K: int, 
    shortcut: int
) -> torch.Tensor:
    """
    each individual is connected to its K immediate neighbors only.
    K=2 by default.
    shortcut is the number of edges changed after connection.
    returns a N*N, each row indicates the indices of neighbours, if not enough neighbour, pending it's own indice.
    """
    dist_matrix = get_distance_matrix(population)
    row_dist_rank = row_argsort(dist_matrix)

    adjacancy_matrix = build_adjacancy_matrix_by_K_nearest_neighbour(row_dist_rank, K)

    adjacancy_matrix = mutate_shortcut(
        adjacancy_matrix=adjacancy_matrix, 
        num_shortcut=shortcut
    )

    return adjacancy_matrix

def get_full_neighbour(population: torch.Tensor) -> torch.Tensor:
    """
    fully connected
    """
    N = population.shape[0]
    adjacancy_matrix = torch.ones((N, N), dtype=torch.int32, device=population.device)
    return adjacancy_matrix

def get_ring_neighbour(population: torch.Tensor, K: int) -> torch.Tensor:
    """
    every individual is connected to two others;
    No more details, we apply same implementation from PyGMO:
    ''The L-best-k topology consists of n nodes arranged in a ring, in which
    node i is connected to each node in {(i+j) mod n : j = +-1,+-2, . . . ,+-k}.'' \n
    [Mohais et al., 2005] http://dx.doi.org/10.1007/11589990_80
    """
    N = population.shape[0]
    adjacancy_matrix = torch.zeros((N, N), dtype=torch.int32, device=population.device)

    for i in range(N):
        for k in range(-K, K + 1):
            j = (i + k) % N
            adjacancy_matrix[i, j] = 1

    return adjacancy_matrix

def get_square_neighbour(population: torch.Tensor) -> torch.Tensor:
    """
    "The square is a graph representing a rectangular lattice that folds like a torus.
    This structure, albeit artificial, is commonly used to represent neighborhoods in 
    the Evolutionary Computation and Cellular Automata communities,
    and is referred to as the von Neumann neighborhood"

    according to PyGMO, reshape N individual into row*col 
    where number rows and cols are as close as possible.
    Each connect to the upper, lower, left and right neighbour
    """
    N = population.shape[0]
    device = population.device

    col = math.floor(math.sqrt(N))
    while col > 1 and N % col != 0:
        col -= 1
    row = N // col
    
    if col <= 2:
        print(
            f"Population size is {N}, when creating square topology, "
            f"{row}*{col} may cause strange topo"
        )

    np_mat = np.arange(N).reshape((col, row))
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    adj_mat = np.zeros((N, N), dtype=np.int32)
    for i in range(col):
        for j in range(row):
            x = np_mat[i, j]
            for dx, dy in directions:
                nx = (i + dx) % col
                ny = (j + dy) % row
                y = np_mat[nx, ny]
                adj_mat[x, y] = 1

    adjacancy_matrix = torch.from_numpy(adj_mat).to(device=device, dtype=torch.int32)
    return adjacancy_matrix

def get_neighbour_best_fitness(
    fitness: torch.Tensor, 
    adjacancy_list: torch.Tensor
):
    """
    given an adjacancy list, and its masking output the neighbour fitness
    """
 
    N, M = adjacancy_list.shape
    expanded_fitness = fitness.unsqueeze(0).expand(N, -1)  # shape(N, N) 
    selected_fitness = torch.gather(expanded_fitness, 1, adjacancy_list)
    min_val, min_indices = torch.min(selected_fitness, dim=1)
    best_indices = torch.gather(adjacancy_list, 1, min_indices.view(-1,1)).view(-1)

    neighbour_best_fitness = min_val
    neighbour_best_indice = best_indices

    return neighbour_best_fitness, neighbour_best_indice

def build_adjacancy_matrix_by_K_nearest_neighbour(
    distance_ranking: torch.Tensor,
    K: int
) -> torch.Tensor:
    """
    build adjacancy matrix by k nearest neighbour.
    input N*N distance_ranking.
    get N*(K+1) adjacancy list (duplicate).
    flatten to get indices.
    form a new adjacancy matrix.
    """
    N = distance_ranking.shape[0]

    row_idx = torch.arange(N, device=distance_ranking.device).view(N, 1)
    row_idx = row_idx.expand(N, K + 1)  # shape (N, K+1)

    top_k_indices = distance_ranking[:, : (K + 1)]
    
    index_X = top_k_indices.reshape(-1)
    row_idx_flat = row_idx.reshape(-1)

    A = torch.zeros((N, N), dtype=torch.int64, device=distance_ranking.device)
    A[row_idx_flat, index_X] = 1
    A[index_X, row_idx_flat] = 1

    return A

def build_adjacancy_list_from_matrix(
    adjacancy_matrix: torch.Tensor,
    keep_self_loop: bool = True
):
    """
    given N*N adjacancy matrix, for every row i, output the outgoing neighbour in length N.
    fill the rest as its own indice.
    output the masking at the same time, indicating the mapping.
    """
    N = adjacancy_matrix.shape[0]

    adjacancy_list = []
    for i in range(N):
        row = adjacancy_matrix[i]
        indices = torch.nonzero(row, as_tuple=True)[0]
        if indices.shape[0] < N:
            pad_len = N - indices.shape[0]
            pad_vals = torch.full((pad_len,), i, dtype=indices.dtype, device=indices.device)
            indices = torch.cat([indices, pad_vals], dim=0)
        adjacancy_list.append(indices[:N].view(1, -1))

    adjacancy_list = torch.cat(adjacancy_list, dim=0)  # shape (N, N)

    adjacancy_list_masking = torch.ones((N, N), dtype=torch.int32, device=adjacancy_matrix.device)
    if not keep_self_loop:
        self_mask = (adjacancy_list == torch.arange(N, device=adjacancy_matrix.device).unsqueeze(1))
        adjacancy_list_masking[self_mask] = 0

    return adjacancy_list, adjacancy_list_masking

def mutate_shortcut(
    adjacancy_matrix: torch.Tensor, 
    num_shortcut: int
) -> torch.Tensor:
    """
    given an N*N adjacancy matrix. (indicating undirected connections)
    change num_shortcut pairs of edge to others.
    """
    N = adjacancy_matrix.shape[0]
    diag_matrix = torch.eye(N, dtype=adjacancy_matrix.dtype, device=adjacancy_matrix.device)

    upper_no_diag = torch.triu(adjacancy_matrix - diag_matrix, diagonal = 1)
    flatten = upper_no_diag.reshape(-1)  # (N*N,) 含 0/1

    rows = torch.arange(N, device=adjacancy_matrix.device).repeat_interleave(N)
    cols = torch.arange(N, device=adjacancy_matrix.device).repeat(N)

    upper_tri_mask = (rows < cols)
    rows = rows[upper_tri_mask]
    cols = cols[upper_tri_mask]

    mutate_mask = select_from_mask(flatten, num_shortcut)

    row_or_col = torch.randint(
        low=0, high=2, size=(rows.shape[0],), device = rows.device
    ).bool()  # True->变 row, False->变 col

    new_nodes = torch.randint(
        low=0, high=N, size=(rows.shape[0],), device=rows.device
    )

    new_rows = torch.where(row_or_col, new_nodes, rows)
    new_cols = torch.where(~row_or_col, new_nodes, cols)

    final_rows = torch.where(mutate_mask.bool(), new_rows, rows)
    final_cols = torch.where(mutate_mask.bool(), new_cols, cols)

    mutated = torch.zeros_like(adjacancy_matrix)
    mutated[final_rows, final_cols] = 1
    mutated[final_cols, final_rows] = 1

    mutated = torch.clamp(mutated + diag_matrix, min=0, max=1)

    return mutated