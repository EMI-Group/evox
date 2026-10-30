"""Multi-objective algorithms of the 2.0 functional API.

Each module is one algorithm: a frozen config dataclass +
frozen state dataclass + plain ``init``/``init_step``/``step`` functions (the
STEP protocol of ``evox_etl.core.algorithm``; no ``final_step`` in this
family). ``step(config, state, evaluate)`` owns ONE full generation: generate
the offspring batch, ``fitness = evaluate(offspring)`` through the workflow's
opaque closure, then merge and run the environmental selection. See
``DESIGN.md`` §4-5 and the per-module docstrings.

Configs are constructed via the module-level ``make_*`` constructors
(``make_nsga2`` ... ``make_hype``), which normalize array-like bounds to flat
float tuples and raise ``ValueError`` on malformed bounds. The ``*Config``
dataclasses remain importable directly and are plain frozen dataclasses.
"""

__all__ = [
    "HypE",
    "HypEConfig",
    "MOEAD",
    "MOEADConfig",
    "NSGA2",
    "NSGA2Config",
    "NSGA3",
    "NSGA3Config",
    "RVEA",
    "RVEAConfig",
    "RVEAa",
    "RVEAaConfig",
    "make_hype",
    "make_moead",
    "make_nsga2",
    "make_nsga3",
    "make_rvea",
    "make_rveaa",
]

from .hype import HypEConfig, make_hype
from .moead import MOEADConfig, make_moead
from .nsga2 import NSGA2Config, make_nsga2
from .nsga3 import NSGA3Config, make_nsga3
from .rvea import RVEAConfig, make_rvea
from .rveaa import RVEAaConfig, make_rveaa

# API compatibility: the 1.x `algorithms/mo/__init__.py` exports the BARE algorithm
# names; the same names are re-exported here (and by the parent
# `evox_etl.algorithms`) as aliases of the matching config dataclasses.  The
# `*Config` names and the `make_*` constructors stay the port-native spelling.
HypE = HypEConfig
MOEAD = MOEADConfig
NSGA2 = NSGA2Config
NSGA3 = NSGA3Config
RVEA = RVEAConfig
RVEAa = RVEAaConfig
