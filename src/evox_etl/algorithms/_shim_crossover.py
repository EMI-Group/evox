"""DEPRECATED test-compat stub — re-exports canonical ``evox_etl.operators.crossover``.

``unit_test/etl/algorithms/test_shim_crossover.py`` (sibling node, outside this
worker's write scope) still imports this module. Once the root agent converts
that test to import the canonical operators directly, DELETE this file.
"""
import etl

from evox_etl.operators.crossover import (
    DE_arithmetic_recombination,
    DE_binary_crossover,
    DE_exponential_crossover,
    simulated_binary,
    simulated_binary_half,
)
from evox_etl.operators.crossover import DE_differential_sum as _de_diff_sum


def DE_differential_sum(
    key, diff_padding_num, num_diff_vectors, index, population, F=None, replace=False
):
    """Canonical DE_differential_sum with the sampled-index return pinned to int32."""
    difference_sum, first = _de_diff_sum(
        key, diff_padding_num, num_diff_vectors, index, population, F=F, replace=replace
    )
    return difference_sum, etl.cast(first, "int32")


__all__ = [
    "DE_arithmetic_recombination", "DE_binary_crossover", "DE_differential_sum",
    "DE_exponential_crossover", "simulated_binary", "simulated_binary_half",
]
