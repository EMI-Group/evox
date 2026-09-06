"""Functional MO algorithms — evox_etl ports of ``torch evox.algorithms.mo``.

Each module mirrors one torch algorithm class as a frozen config dataclass +
frozen state dataclass + plain ``init``/``init_ask``/``init_tell``/``ask``/
``tell`` functions (see ``DESIGN.md`` §4-5 and the per-module docstrings).

Configs are constructed via the module-level ``make_*`` constructors
(``make_nsga2`` ... ``make_hype``), which normalize array-like bounds to flat
float tuples and raise ``ValueError`` on malformed bounds. The ``*Config``
dataclasses remain importable for parity and are plain frozen dataclasses.
"""

__all__ = [
    "HypEConfig",
    "MOEADConfig",
    "NSGA2Config",
    "NSGA3Config",
    "RVEAConfig",
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
