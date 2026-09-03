"""Functional port of all torch evox algorithms (evox_etl).

Mirrors torch evox.algorithms exports: frozen config dataclasses per algorithm
under the torch class names (ES configs are aliased from their ``*Config``
names). All algorithms follow the binding functional contract: ``init(config,
key) -> State``, ``ask(config, state) -> (candidates, new_state)``, ``tell(
config, state, fitness) -> new_state`` (plus ``init_ask``/``init_tell`` for
algorithms with a full-population first generation). See
``src/evox_etl/DESIGN.md`` §4-5.
"""
from .mo import (
    HypEConfig,
    MOEADConfig,
    NSGA2Config,
    NSGA3Config,
    RVEAConfig,
    RVEAaConfig,
)
from .so import (
    ARS,
    ASEBO,
    CLPSO,
    CMAES,
    CSO,
    DE,
    DES,
    DMSPSOEL,
    ESMC,
    FSPSO,
    GuidedES,
    JaDE,
    NoiseReuseES,
    ODE,
    OpenES,
    PSO,
    PersistentES,
    SHADE,
    SLPSOGS,
    SLPSOUS,
    SNES,
    SeparableNES,
    XNES,
    CoDE,
    SaDE,
)

# torch-style bare-name aliases for the MO family configs
NSGA2 = NSGA2Config
NSGA3 = NSGA3Config
MOEAD = MOEADConfig
RVEA = RVEAConfig
RVEAa = RVEAaConfig
HypE = HypEConfig

__all__ = [
    # DE Variants
    "DE",
    "SHADE",
    "CoDE",
    "SaDE",
    "ODE",
    "JaDE",
    # ES Variants
    "OpenES",
    "XNES",
    "SeparableNES",
    "DES",
    "SNES",
    "ARS",
    "ASEBO",
    "PersistentES",
    "NoiseReuseES",
    "GuidedES",
    "ESMC",
    "CMAES",
    # PSO Variants
    "CLPSO",
    "CSO",
    "DMSPSOEL",
    "FSPSO",
    "PSO",
    "SLPSOGS",
    "SLPSOUS",
    # MOEAs
    "RVEA",
    "MOEAD",
    "NSGA2",
    "NSGA3",
    "HypE",
    "RVEAa",
]
