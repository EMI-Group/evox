"""evox_etl.utils — functional helpers for the ``evoxc``-based EvoX rewrite.

Plain (non-defn) functions, callable only inside an active evoxc trace (except
``parse_opt_direction``/``compose``, which are pure Python).  Mirrors the JAX
evox v0.9.0 ``utils/common.py`` surface.  See DESIGN.md §4.3.

The torch-1.x ``evox.utils`` jit-fix helpers (``switch``/``clamp*``/``clip``/
``maximum*``/``minimum*``/``lexsort``/``nanmin``/``nanmax``/``randint``) are
re-exported here so ``from evox_etl.utils import clamp`` keeps working; their
CANONICAL home is ``evox_etl.operators.jit_fix_operator`` (import from there in
new code).  The torch-only helpers with no port counterpart
(``VmapInfo``/``register_vmap_op``/``ParamsAndVector``) are not re-exported.
"""

from ..operators.jit_fix_operator import (
    clamp,
    clamp_float,
    clamp_int,
    clip,
    lexsort,
    maximum,
    maximum_float,
    maximum_int,
    minimum,
    minimum_float,
    minimum_int,
    nanmax,
    nanmin,
    randint,
    switch,
)
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
    # jit-fix helpers (canonical home: evox_etl.operators.jit_fix_operator)
    "switch",
    "clamp",
    "clamp_int",
    "clamp_float",
    "clip",
    "maximum",
    "minimum",
    "maximum_float",
    "minimum_float",
    "maximum_int",
    "minimum_int",
    "lexsort",
    "nanmin",
    "nanmax",
    "randint",
]
