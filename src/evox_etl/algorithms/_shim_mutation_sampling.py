"""DEPRECATED test-compat stub — canonical mutation/sampling with the old
polynomial_mutation boundary signature.

``unit_test/etl/algorithms/test_shim_mutation_sampling.py`` (sibling node,
outside this worker's write scope) still imports this module. Once the root
agent converts that test to the canonical `(lb, ub)` signature, DELETE this file.
"""
from evox_etl.operators.mutation import polynomial_mutation as _canonical_pm
from evox_etl.operators.sampling import uniform_sampling


def polynomial_mutation(key, x, boundary, pro_m=1, dis_m=20):
    """Compat wrapper: splits the stacked (2, dim) boundary into (lb, ub)."""
    return _canonical_pm(key, x, boundary[0], boundary[1], pro_m=pro_m, dis_m=dis_m)


__all__ = ["polynomial_mutation", "uniform_sampling"]
