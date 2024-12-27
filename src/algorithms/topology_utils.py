import torch
import math
import numpy as np

from ..utils import clamp

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

def get_square_neighbour(population: torch.Tensor):
        """
        Constructs a square topology for the population, where each individual connects
        to its upper, lower, left, and right neighbors in a toroidal grid.

        Args:
            population (torch.Tensor): Population tensor of shape (N, ...), where N is the number of individuals.

        Returns:
            torch.Tensor: Adjacency matrix of shape (N, N) representing the neighborhood connections.
        """
        N = population.shape[0]
        # Calculate the number of columns and rows for the toroidal grid
        col = int(torch.floor(torch.sqrt(torch.tensor(N, dtype=torch.float32))).item())
        while col > 1 and N % col != 0:
            col -= 1
        row = (N // col)

        # Warn if the topology is degenerate (e.g., ring topology)
        if col <= 2:
            print(
                f"Population size is {N}. When creating square topology, number of rows and cols {row}x{col} "
                f"may cause unusual topology."
            )

        # Create a 2D grid of indices
        grid_indices = torch.arange(N).reshape(col, row)

        # Define neighborhood directions (right, down, left, up)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        # Initialize the adjacency matrix
        adjacancy_matrix = torch.zeros((N, N), dtype=torch.float32)

        # Iterate over the grid and connect neighbors
        for i in range(col):
            for j in range(row):
                x = grid_indices[i, j]
                for di, dj in directions:
                    ni, nj = (i + di) % col, (j + dj) % row
                    y = grid_indices[ni, nj]
                    adjacancy_matrix[x, y] = 1

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
    Given N x N adjacency matrix, for every row i, output the outgoing neighbors in length N.
    Fill the rest as its own index.
    Output the masking at the same time, indicating the mapping.
    
    Args:
        adjacancy_matrix (torch.Tensor): Input adjacency matrix of shape (N, N).
        keep_self_loop (bool): Whether to keep self-loops in the adjacency list.
        
    Returns:
        tuple: (adjacancy_list, adjacancy_list_masking)
            - adjacancy_list (torch.Tensor): Tensor of shape (N, N) with neighbor indices.
            - adjacancy_list_masking (torch.Tensor): Tensor of shape (N, N) indicating the mapping (1 for valid, 0 for padding).
    """
    N = adjacancy_matrix.shape[0]

    # Initialize masking with all ones
    adjacancy_list_masking = torch.ones((N, N), dtype=torch.float32, device=adjacancy_matrix.device)

    # Get the row indices of non-zero elements for each row
    adjacancy_list = torch.stack([get_row_indices(adjacancy_matrix[i], N) for i in range(N)])

    # Update masking to indicate valid indices
    adjacancy_list_masking = torch.where(adjacancy_list == -1, 0, adjacancy_list_masking)

    # Identity matrix for self-loops
    row_indices = torch.arange(N, dtype=torch.int64, device=adjacancy_matrix.device)
    identity = row_indices.unsqueeze(0).repeat(N, 1)

    if not keep_self_loop:
        adjacancy_list_masking = torch.where(
            adjacancy_list == identity, 0, adjacancy_list_masking
        )

    # Replace -1 with self-loop indices
    adjacancy_list = torch.where(adjacancy_list == -1, identity, adjacancy_list)

    return adjacancy_list, adjacancy_list_masking

def get_row_indices(row: torch.Tensor, N: int) -> torch.Tensor:
    nonzero_indices = torch.nonzero(row).flatten()
    if len(nonzero_indices) < N:
        # Pad with -1 to maintain consistent length
        padding = -torch.ones(N - len(nonzero_indices), dtype=torch.int64, device=row.device)
        return torch.cat([nonzero_indices, padding], dim=0)
    return nonzero_indices    

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

    # upper_tri_mask = (rows < cols)
    # rows = rows[upper_tri_mask]
    # cols = cols[upper_tri_mask]

    mutate_mask = select_from_mask(flatten, num_shortcut)

    row_or_col = torch.randint(
        low=0, high=2, size=(rows.shape[0],), device=rows.device
    ).to(dtype=torch.bool)

    new_nodes = torch.randint(
        low=0, high=N, size=(rows.shape[0],), device=rows.device
    )

    new_rows = torch.where(row_or_col, new_nodes, rows)
    new_cols = torch.where(~row_or_col, new_nodes, cols)

    final_rows = torch.where(mutate_mask, new_rows, rows)
    final_cols = torch.where(mutate_mask, new_cols, cols)

    mutated = torch.zeros_like(adjacancy_matrix)
    mutated[final_rows, final_cols] = 1
    mutated[final_cols, final_rows] = 1

    mutated = clamp(mutated + diag_matrix, torch.as_tensor(0), torch.as_tensor(1)) # TODO

    return mutated