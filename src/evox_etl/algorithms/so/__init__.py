"""SO algorithm families — mirror of torch evox.algorithms.so exports.

Note: the es_variants family names its frozen config dataclasses with a
``*Config`` suffix; this module re-exports them under their torch-style bare
names so the API surface mirrors torch evox.
"""
from .de_variants import DE, SHADE, CoDE, SaDE, ODE, JaDE
from .es_variants import (
    ARSConfig,
    ASEBOConfig,
    CMAESConfig,
    DESConfig,
    ESMCConfig,
    GuidedESConfig,
    NoiseReuseESConfig,
    OpenESConfig,
    PersistentESConfig,
    SeparableNESConfig,
    SNESConfig,
    XNESConfig,
)
from .pso_variants import CLPSO, CSO, DMSPSOEL, FSPSO, PSO, SLPSOGS, SLPSOUS

# torch-style bare-name aliases for the ES family
OpenES = OpenESConfig
XNES = XNESConfig
SeparableNES = SeparableNESConfig
DES = DESConfig
SNES = SNESConfig
ARS = ARSConfig
ASEBO = ASEBOConfig
PersistentES = PersistentESConfig
NoiseReuseES = NoiseReuseESConfig
GuidedES = GuidedESConfig
ESMC = ESMCConfig
CMAES = CMAESConfig

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
]
