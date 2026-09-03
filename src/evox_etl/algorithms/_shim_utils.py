"""DEPRECATED test-compat stub — re-exports ``_jit_fix_operator``.

``unit_test/etl/algorithms/test_shim_utils.py`` (sibling node, outside this
worker's write scope) still imports this module. It re-exports the canonical
staging module ``evox_etl.algorithms._jit_fix_operator`` (a move-ready port of
torch ``evox.utils.jit_fix_operator``). Once the root agent converts that test
to import ``evox_etl.operators.jit_fix_operator`` directly, DELETE this file.
"""
from ._jit_fix_operator import (
    clamp, clamp_float, clamp_int, lexsort, maximum, maximum_int,
    minimum, minimum_int, nanmax, nanmin, randint,
)

__all__ = [
    "clamp", "clamp_float", "clamp_int", "lexsort", "maximum", "maximum_int",
    "minimum", "minimum_int", "nanmax", "nanmin", "randint",
]
