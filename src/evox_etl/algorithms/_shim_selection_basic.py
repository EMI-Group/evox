"""DEPRECATED test-compat stub — canonical basic selection, shim key-split convention.

``unit_test/etl/algorithms/test_shim_selection_basic.py`` (sibling node, outside
this worker's write scope) pins the draw stream of the old shim, which split the
key once before drawing. Wrap the canonical operators with the same split so the
streams match. Once the root agent converts that test, DELETE this file.
"""
import etl.random as random

from evox_etl.operators.selection import (
    select_rand_pbest as _canonical_pbest,
    tournament_selection as _canonical_ts,
    tournament_selection_multifit as _canonical_tsm,
)

from ..operators.jit_fix_operator import _take_along_axis  # noqa: F401  (kept for old importers)


def tournament_selection(key, n_round, fitness, tournament_size=2):
    key_rand, _ = random.split(key)
    return _canonical_ts(key_rand, n_round, fitness, tournament_size)


def tournament_selection_multifit(key, n_round, fitnesses, tournament_size=2):
    key_rand, _ = random.split(key)
    return _canonical_tsm(key_rand, n_round, fitnesses, tournament_size)


def select_rand_pbest(key, percent, population, fitness):
    key_rand, _ = random.split(key)
    return _canonical_pbest(key_rand, percent, population, fitness)


__all__ = [
    "select_rand_pbest", "tournament_selection", "tournament_selection_multifit",
    "_take_along_axis",
]
