"""DEPRECATED test-compat stub — re-exports canonical non-dominated selection.

``unit_test/etl/algorithms/test_shim_selection_nd.py`` (sibling node, outside
this worker's write scope) loads this file by path. Once the root agent
converts that test to import the canonical operators directly, DELETE this file.
"""
from evox_etl.operators.selection.non_dominate import (
    crowding_distance,
    dominate_relation,
    nd_environmental_selection,
    non_dominate_rank,
)

__all__ = [
    "dominate_relation", "non_dominate_rank",
    "crowding_distance", "nd_environmental_selection",
]
