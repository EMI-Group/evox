"""Single import path for operator/util shims used by algorithm ports.

The evox_etl.operators / evox_etl.utils packages are being built in parallel.
Until they land, algorithm code imports every operator/util function from here.
Each name below is a PLAIN function (callable only inside an active etl trace)
ported 1:1 from the torch reference (see the individual _shim_* modules for
sources and semantics notes). Signatures follow DESIGN.md §4.3 (key-first for
random functions).
"""
from ._shim_utils import (
    clamp,
    clamp_float,
    clamp_int,
    lexsort,
    maximum,
    maximum_int,
    minimum,
    minimum_int,
    nanmax,
    nanmin,
    randint,
)
from ._shim_crossover import (
    DE_arithmetic_recombination,
    DE_binary_crossover,
    DE_differential_sum,
    DE_exponential_crossover,
    simulated_binary,
    simulated_binary_half,
)
from ._shim_mutation_sampling import polynomial_mutation, uniform_sampling
from ._shim_selection_basic import (
    select_rand_pbest,
    tournament_selection,
    tournament_selection_multifit,
)
from ._shim_selection_nd import (
    crowding_distance,
    dominate_relation,
    nd_environmental_selection,
    non_dominate_rank,
)
from ._shim_selection_rvea import apd_fn, ref_vec_guided

__all__ = [
    "clamp",
    "clamp_float",
    "clamp_int",
    "lexsort",
    "maximum",
    "maximum_int",
    "minimum",
    "minimum_int",
    "nanmax",
    "nanmin",
    "randint",
    "DE_arithmetic_recombination",
    "DE_binary_crossover",
    "DE_differential_sum",
    "DE_exponential_crossover",
    "simulated_binary",
    "simulated_binary_half",
    "polynomial_mutation",
    "uniform_sampling",
    "select_rand_pbest",
    "tournament_selection",
    "tournament_selection_multifit",
    "crowding_distance",
    "dominate_relation",
    "nd_environmental_selection",
    "non_dominate_rank",
    "apd_fn",
    "ref_vec_guided",
]
