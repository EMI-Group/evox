"""evox_etl.utils — functional helpers for the ETL-based EvoX rewrite.

Plain (non-defn) functions, callable only inside an active etl trace (except
``parse_opt_direction``/``compose``, which are pure Python).  Mirrors the JAX
evox v0.9.0 ``utils/common.py`` surface.  See DESIGN.md §4.3.
"""

from .common import (
    cal_max,
    chebyshev_dist,
    compose,
    cos_dist,
    euclidean_dist,
    manhattan_dist,
    min_by,
    pair_max,
    pairwise_chebyshev_dist,
    pairwise_euclidean_dist,
    pairwise_func,
    pairwise_manhattan_dist,
    parse_opt_direction,
    rank,
    rank_based_fitness,
    tree_flatten,
    tree_leaves,
    tree_map,
    tree_unflatten,
)

__all__ = [
    "min_by",
    "euclidean_dist",
    "manhattan_dist",
    "chebyshev_dist",
    "pairwise_func",
    "pairwise_euclidean_dist",
    "pairwise_manhattan_dist",
    "pairwise_chebyshev_dist",
    "pair_max",
    "cos_dist",
    "cal_max",
    "compose",
    "rank",
    "rank_based_fitness",
    "parse_opt_direction",
    "tree_flatten",
    "tree_leaves",
    "tree_map",
    "tree_unflatten",
]
