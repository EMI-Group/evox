import torch

from ...core import ModuleBase, jit_class, trace_impl
from ...utils import TracingWhile, lexsort


def dominate_relation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return a matrix A, where A_{ij} is True if x_i dominates y_j."""
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
    count_desc = torch.sum(pareto_front[:, None] * dominate_relation_matrix, dim=0)

    # Update dominate_count (remove those in the current Pareto front)
    dominate_count = dominate_count - count_desc
    dominate_count = dominate_count - pareto_front.int()

    return rank, dominate_count


def _non_dominated_sort_script(x: torch.Tensor) -> torch.Tensor:
    """
    Perform non-dominated sort using PyTorch in torch.script mode.

    :param x: An array with shape (n, m) where n is the population size and m is the number of objectives.

    :returns:
        A one-dimensional tensor representing the ranking, starting from 0.
    """

    n, m = x.size()

    # Domination relation matrix (n x n)
    dominate_relation_matrix = dominate_relation(x, x)

    # Count how many times each individual is dominated
    dominate_count = dominate_relation_matrix.sum(dim=0)

    # Initialize rank array
    rank = torch.zeros(n, dtype=torch.int32, device=x.device)
    current_rank = 0

    # Identify individuals in the first Pareto front (those that are not dominated)
    pareto_front = dominate_count == 0

    # Iteratively identify Pareto fronts
    while pareto_front.any():
        rank, dominate_count = update_dc_and_rank(dominate_relation_matrix, dominate_count, pareto_front, rank, current_rank)
        current_rank += 1
        pareto_front = dominate_count == 0

    return rank


_NDS_cache = None


@jit_class
class NonDominatedSort(ModuleBase):
    """
    A module for performing non-dominated sorting, implementing caching and support for PyTorch's full map-reduce method.

    This class provides an efficient implementation of non-dominated sorting using both direct computation and a
    traceable map-reduce method for large-scale multi-objective optimization problems.
    """

    def __new__(cls):
        global _NDS_cache
        if _NDS_cache is not None:
            return _NDS_cache
        else:
            return super().__new__(cls)

    def __init__(self):
        """
        Initialize the NonDominatedSort module, setting up caching for efficient reuse.
        """
        global _NDS_cache
        if _NDS_cache is not None:
            return
        super().__init__()
        _NDS_cache = self

    def non_dominated_sort(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform non-dominated sorting on the input tensor.
        """
        return _non_dominated_sort_script(x)

    @trace_impl(non_dominated_sort)
    def trace_non_dominated_sort(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform non-dominated sorting using PyTorch's full map-reduce method for efficient computation.

        :param x: An array with shape (n, m) where n is the population size and m is the number of objectives.

        :returns: A one-dimensional tensor representing the ranking, starting from 0.
        """
        n, m = x.size()

        # Domination relation matrix (n x n)
        dominate_relation_matrix = dominate_relation(x, x)

        # Count how many times each individual is dominated
        dominate_count = dominate_relation_matrix.sum(dim=0)

        # Initialize rank array
        rank = torch.zeros(n, dtype=torch.int32, device=x.device)
        current_rank = torch.tensor(0, dtype=torch.int32, device=x.device)

        # Identify individuals in the first Pareto front (those that are not dominated)
        pareto_front = dominate_count == 0

        def body_func(
            rank: torch.Tensor,
            dominate_count: torch.Tensor,
            current_rank: torch.IntTensor,
            pareto_front: torch.BoolTensor,
            dominate_relation_matrix: torch.Tensor,
        ):
            # Update rank and dominate count
            rank, dominate_count = update_dc_and_rank(
                dominate_relation_matrix, dominate_count, pareto_front, rank, current_rank
            )

            # Move to next rank
            current_rank = current_rank + 1
            pareto_front = dominate_count == 0

            return rank, dominate_count, current_rank, pareto_front, dominate_relation_matrix

        if not hasattr(self, "_while_loop_"):
            self._while_loop_ = TracingWhile(lambda x, y, p, q, a: q.any(), body_func)
        rank, _, _, _, _ = self._while_loop_.loop(rank, dominate_count, current_rank, pareto_front, dominate_relation_matrix)

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
    distance_range = costs[num_valid_elem - 1, :] - costs[0, :]
    distance = torch.empty(costs.size(), device=costs.device)
    distance = distance.scatter(0, rank[1:-1], (costs[2:] - costs[:-2]) / distance_range)
    distance[rank[0], :] = torch.inf
    distance[rank[num_valid_elem - 1], :] = torch.inf
    crowding_distances = torch.where(mask.unsqueeze(1), distance, -torch.inf)
    crowding_distances = torch.sum(crowding_distances, dim=1)

    return crowding_distances


def _non_dominate_rank(f: torch.Tensor):
    rank = NonDominatedSort().non_dominated_sort(f)
    return rank


_non_dominate_rank.__prepare_scriptable__ = lambda: _non_dominated_sort_script


def nd_environmental_selection(x: torch.Tensor, f: torch.Tensor, topk: int):
    """
    Perform environmental selection using the non-dominated sorting and crowding distance.

    :param x: A 2D tensor where each row represents a solution, and each column represents a variable.
    :param f: A 2D tensor where each row represents a solution, and each column represents an objective.
    :param topk: The number of solutions to select.

    :returns:
        A tuple of four tensors. The first tensor contains the selected solutions, the second tensor contains the corresponding objective values, the third tensor contains the non-dominated sorting rank of the selected solutions, and the fourth tensor contains the crowding distance of the selected solutions.
    """

    rank = _non_dominate_rank(f)
    order = torch.argsort(rank, stable=True)
    worst_rank = rank[order[topk - 1]]
    mask = rank == worst_rank
    crowding_dis = crowding_distance(f, mask)
    dis_order = torch.argsort(crowding_dis, stable=True)
    combined_order = lexsort([-dis_order, rank])[:topk]
    return x[combined_order], f[combined_order], rank[combined_order], crowding_dis[combined_order]
