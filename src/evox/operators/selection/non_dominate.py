import torch

from evox.core import compile
from evox.utils import lexsort, register_vmap_op


def dominate_relation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return the domination relation matrix A, where A_{ij} is True if x_i dominates y_j.

    :param x: An array with shape (n1, m) where n1 is the population size and m is the number of objectives.
    :param y: An array with shape (n2, m) where n2 is the population size and m is the number of objectives.

    :returns: The domination relation matrix of x and y.
    """
    # Expand the dimensions of x and y so that we can perform element-wise comparisons
    # Add new dimensions to x and y to prepare them for broadcasting
    x_expanded = x.unsqueeze(1)  # Shape (n1, 1, m)
    y_expanded = y.unsqueeze(0)  # Shape (1, n2, m)

    # Broadcasted comparison: each pair (x_i, y_j)
    less_than_equal = x_expanded <= y_expanded  # Shape (n1, n2, m)
    strictly_less_than = x_expanded < y_expanded  # Shape (n1, n2, m)

    # Check the domination condition: x_i dominates y_j
    domination_matrix = less_than_equal.all(dim=2) & strictly_less_than.any(dim=2)

    return domination_matrix


def update_dc_and_rank(
    dominate_relation_matrix: torch.Tensor,
    dominate_count: torch.Tensor,
    pareto_front: torch.BoolTensor,
    rank: torch.Tensor,
    current_rank: int,
):
    """
    Update the dominate count and ranks for the current Pareto front.

    :param dominate_relation_matrix: The domination relation matrix between individuals.
    :param dominate_count: The count of how many individuals dominate each individual.
    :param pareto_front: A tensor indicating which individuals are in the current Pareto front.
    :param rank: A tensor storing the rank of each individual.
    :param current_rank: The current Pareto front rank.

    :returns:
        - **rank**: Updated rank tensor.
        - **dominate_count**: Updated dominate count tensor.
    """

    # Update the rank for individuals in the Pareto front
    rank = torch.where(pareto_front, current_rank, rank)
    # Calculate how many individuals in the Pareto front dominate others
    count_desc = torch.sum(pareto_front.unsqueeze(-1) * dominate_relation_matrix, dim=-2)

    # Update dominate_count (remove those in the current Pareto front)
    dominate_count = dominate_count - count_desc
    dominate_count = dominate_count - pareto_front.int()

    return rank, dominate_count


_compiled_update_dc_and_rank = compile(update_dc_and_rank, fullgraph=True)


def _igr_fake(
    dominate_relation_matrix: torch.Tensor,
    dominate_count: torch.Tensor,
    rank: torch.Tensor,
    pareto_front: torch.Tensor,
) -> torch.Tensor:
    return rank.new_empty(dominate_count.size())


def _igr_fake_vmap(
    dominate_relation_matrix: torch.Tensor,
    dominate_count: torch.Tensor,
    rank: torch.Tensor,
    pareto_front: torch.Tensor,
) -> torch.Tensor:
    return rank.new_empty(dominate_count.size())


def _vmap_iterative_get_ranks(
    dominate_relation_matrix: torch.Tensor,
    dominate_count: torch.Tensor,
    rank: torch.Tensor,
    pareto_front: torch.Tensor,
) -> torch.Tensor:
    current_rank = 0
    while pareto_front.any():
        rank, dominate_count = (_compiled_update_dc_and_rank if torch.compiler.is_compiling() else update_dc_and_rank)(
            dominate_relation_matrix, dominate_count, pareto_front, rank, current_rank
        )
        current_rank += 1
        new_pareto_front = dominate_count == 0
        pareto_front = torch.where(pareto_front.any(dim=-1, keepdim=True), new_pareto_front, pareto_front)
    return rank


@register_vmap_op(fake_fn=_igr_fake, vmap_fn=_vmap_iterative_get_ranks, fake_vmap_fn=_igr_fake_vmap, max_vmap_level=2)
def _iterative_get_ranks(
    dominate_relation_matrix: torch.Tensor,
    dominate_count: torch.Tensor,
    rank: torch.Tensor,
    pareto_front: torch.Tensor,
) -> torch.Tensor:
    current_rank = 0
    while pareto_front.any():
        rank, dominate_count = (_compiled_update_dc_and_rank if torch.compiler.is_compiling() else update_dc_and_rank)(
            dominate_relation_matrix, dominate_count, pareto_front, rank, current_rank
        )
        current_rank += 1
        pareto_front = dominate_count == 0
    return rank


def non_dominate_rank(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the non-domination rank for a set of solutions in multi-objective optimization.

    The non-domination rank is a measure of the Pareto optimality of each solution.

    :param f: A 2D tensor where each row represents a solution, and each column represents an objective.

    :returns:
        A 1D tensor containing the non-domination rank for each solution.
    """

    n = x.size(0)
    # Domination relation matrix (n x n)
    dominate_relation_matrix = dominate_relation(x, x)
    # Count how many times each individual is dominated
    dominate_count = dominate_relation_matrix.sum(dim=0)
    # Initialize rank array
    rank = torch.zeros(n, dtype=torch.int32, device=x.device)
    # Identify individuals in the first Pareto front (those that are not dominated)
    pareto_front = dominate_count == 0
    # Iteratively identify Pareto fronts
    rank = _iterative_get_ranks(dominate_relation_matrix, dominate_count, rank, pareto_front)
    return rank


def crowding_distance(costs: torch.Tensor, mask: torch.Tensor):
    """
    Compute the crowding distance for a set of solutions in multi-objective optimization.

    The crowding distance is a measure of the diversity of solutions within a Pareto front.

    :param costs: A 2D tensor where each row represents a solution, and each column represents an objective.
    :param mask: A 1D boolean tensor indicating which solutions should be considered.

    :returns:
        A 1D tensor containing the crowding distance for each solution.
    """
    total_len = costs.size(0)
    if mask is None:
        num_valid_elem = total_len
        mask = torch.ones(total_len, dtype=torch.bool)
    else:
        num_valid_elem = mask.sum()

    inverted_mask = ~mask

    inverted_mask = inverted_mask.unsqueeze(1).expand(-1, costs.size(1)).to(costs.dtype)

    rank = lexsort([costs, inverted_mask], dim=0)
    costs = torch.gather(costs, dim=0, index=rank)
    distance_range = costs[num_valid_elem - 1] - costs[0]
    distance = torch.empty(costs.size(), device=costs.device)
    distance = distance.scatter(0, rank[1:-1], (costs[2:] - costs[:-2]) / distance_range)
    distance[rank[0], :] = torch.inf
    distance[rank[num_valid_elem - 1], :] = torch.inf
    crowding_distances = torch.where(mask.unsqueeze(1), distance, -torch.inf)
    crowding_distances = torch.sum(crowding_distances, dim=1)

    return crowding_distances


def nd_environmental_selection(x: torch.Tensor, f: torch.Tensor, topk: int):
    """
    Perform environmental selection based on non-domination rank and crowding distance.

    :param x: A 2D tensor where each row represents a solution, and each column represents a decision variable.
    :param f: A 2D tensor where each row represents a solution, and each column represents an objective.
    :param topk: The number of solutions to select.

    :returns:
        A tuple of four tensors:
        - **x**: The selected solutions.
        - **f**: The corresponding objective values.
        - **rank**: The non-domination rank of the selected solutions.
        - **crowding_dis**: The crowding distance of the selected solutions.
    """
    rank = non_dominate_rank(f)
    worst_rank = torch.topk(rank, topk, largest=False)[0][-1]
    mask = rank == worst_rank
    crowding_dis = crowding_distance(f, mask)
    combined_order = lexsort([-crowding_dis, rank])[:topk]
    return x[combined_order], f[combined_order], rank[combined_order], crowding_dis[combined_order]
