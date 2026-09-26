"""Non-domination ranking, crowding distance, and NSGA-II environmental selection."""

import etl
import etl.numpy as enp


def _take_along_axis(x, indices, axis):
    """Take entries of ``x`` along ``axis`` at ``indices`` (torch ``take_along_dim``).

    Result shape equals ``indices.shape`` (``x`` must be 1D or 2D).
    """
    if len(x.shape) == 1:
        return etl.gather(x, etl.cast(indices, 'int64'), axis=0)
    if axis == 0:
        m = x.shape[1]
        idx_flat = etl.cast(indices, 'int64') * m + etl.reshape(enp.arange(m, dtype='int64'), (1, m))
        return etl.gather(etl.reshape(x, (-1,)), idx_flat)
    if axis == 1:
        s = x.shape[1]
        idx_flat = etl.reshape(enp.arange(x.shape[0], dtype='int64'), (x.shape[0], 1)) * s + etl.cast(indices, 'int64')
        return etl.gather(etl.reshape(x, (-1,)), idx_flat)
    raise ValueError(f"_take_along_axis: axis {axis} not supported")


def _lexsort(keys, axis=0):
    """Lexicographic sort over ``keys`` (numpy semantics: the LAST key is primary).

    Port of evox.utils.lexsort: stable radix-style multi-key argsort.
    """
    indices = etl.argsort(keys[0], axis=axis, stable=True)
    for key in keys[1:]:
        sorted_key = _take_along_axis(key, indices, axis)
        final_sorted_indices = etl.argsort(sorted_key, axis=axis, stable=True)
        indices = _take_along_axis(indices, final_sorted_indices, axis)
    return indices


def dominate_relation(x, y):
    """Return the domination relation matrix A, where A_{ij} is True if x_i dominates y_j.

    :param x: An array with shape (n1, m) where n1 is the population size and m is the number of objectives.
    :param y: An array with shape (n2, m) where n2 is the population size and m is the number of objectives.

    :returns: The domination relation matrix of x and y.
    """
    n1, m = x.shape[0], x.shape[1]
    n2 = y.shape[0]
    x_expanded = etl.reshape(x, (n1, 1, m))
    y_expanded = etl.reshape(y, (1, n2, m))

    less_than_equal = x_expanded <= y_expanded
    strictly_less_than = x_expanded < y_expanded

    domination_matrix = etl.logical_and(
        enp.sum(etl.cast(less_than_equal, 'int32'), axis=2) == m,
        enp.sum(etl.cast(strictly_less_than, 'int32'), axis=2) > 0,
    )

    return domination_matrix


def non_dominate_rank(x):
    """
    Compute the non-domination rank for a set of solutions in multi-objective optimization.

    The non-domination rank is a measure of the Pareto optimality of each solution.
    Ranks are 0-based: the first Pareto front has rank 0.

    :param f: A 2D tensor where each row represents a solution, and each column represents an objective.

    :returns:
        A 1D tensor containing the non-domination rank for each solution.
    """
    n = x.shape[0]
    # Domination relation matrix (n x n)
    dominate_relation_matrix = dominate_relation(x, x)
    # Count how many times each individual is dominated
    dominate_count = enp.sum(etl.cast(dominate_relation_matrix, 'int32'), axis=0)
    # Initialize rank array
    rank = etl.cast(enp.arange(n, dtype='int64') * 0, 'int32')
    # Identify individuals in the first Pareto front (those that are not dominated)
    pareto_front = dominate_count == 0

    current_rank = etl.constant(etl.core.tensor(0, dtype='int32'))

    def cond_fn(carried):
        r, cr, dc, pf = carried
        return enp.sum(etl.cast(pf, 'int32')) > 0

    def body_fn(carried):
        r, cr, dc, pf = carried
        # Update the rank for individuals in the Pareto front
        r = etl.select(pf, cr, r)
        # Calculate how many individuals in the Pareto front dominate others
        count_desc = enp.sum(
            etl.reshape(etl.cast(pf, 'int32'), (n, 1)) * etl.cast(dominate_relation_matrix, 'int32'),
            axis=0,
        )
        # Update dominate_count (remove those in the current Pareto front)
        dc = dc - count_desc - etl.cast(pf, 'int32')
        cr = etl.cast(cr + 1, 'int32')
        new_pareto_front = dc == 0
        return r, cr, dc, new_pareto_front

    rank, *_ = etl.while_loop(cond_fn, body_fn, (rank, current_rank, dominate_count, pareto_front))
    return rank


def crowding_distance(costs, mask):
    """
    Compute the crowding distance for a set of solutions in multi-objective optimization.

    The crowding distance is a measure of the diversity of solutions within a Pareto front.

    :param costs: A 2D tensor where each row represents a solution, and each column represents an objective.
    :param mask: A 1D boolean tensor indicating which solutions should be considered (None means all).

    :returns:
        A 1D tensor containing the crowding distance for each solution.
    """
    total_len = costs.shape[0]
    if mask is None:
        mask = etl.constant(etl.core.tensor([True] * total_len))
    num_valid_elem = enp.sum(etl.cast(mask, 'int32'))

    inverted_mask = etl.broadcast(
        etl.cast(etl.reshape(etl.logical_not(mask), (total_len, 1)), costs.dtype),
        (total_len, costs.shape[1]),
    )

    rank = _lexsort([costs, inverted_mask], axis=0)
    costs = _take_along_axis(costs, rank, 0)
    c0 = etl.constant(etl.core.tensor(0, dtype='int64'))
    c1 = num_valid_elem - 1
    distance_range = etl.gather(costs, c1, axis=0) - etl.gather(costs, c0, axis=0)

    m = costs.shape[1]
    d_flat_init = etl.constant(etl.core.tensor([0.0] * (total_len * m), dtype=costs.dtype))
    src = (costs[2:] - costs[:-2]) / distance_range
    idx_flat = etl.reshape(
        etl.cast(rank[1:-1], 'int64') * m + etl.reshape(enp.arange(m, dtype='int64'), (1, m)),
        (-1,),
    )
    d_flat = etl.scatter(d_flat_init, idx_flat, etl.reshape(src, (-1,)), axis=0)
    distance = etl.reshape(d_flat, (total_len, m))

    rowmask0 = etl.reshape(enp.arange(total_len, dtype='int64'), (total_len, 1)) == etl.gather(rank, c0, axis=0)
    rowmask1 = etl.reshape(enp.arange(total_len, dtype='int64'), (total_len, 1)) == etl.gather(rank, c1, axis=0)
    distance = etl.select(etl.logical_or(rowmask0, rowmask1), float('inf'), distance)

    crowding_distances = enp.sum(etl.select(etl.reshape(mask, (total_len, 1)), distance, float('-inf')), axis=1)

    return crowding_distances


def nd_environmental_selection(x, f, topk):
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
    vals, _ = etl.topk(rank, topk, largest=False)
    worst_rank = vals[topk - 1]
    mask = rank == worst_rank
    crowding_dis = crowding_distance(f, mask)
    combined_order = _lexsort([-crowding_dis, rank], axis=0)[:topk]
    return (
        etl.gather(x, combined_order, axis=0),
        etl.gather(f, combined_order, axis=0),
        etl.gather(rank, combined_order, axis=0),
        etl.gather(crowding_dis, combined_order, axis=0),
    )
