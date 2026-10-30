"""SO algorithm families of the 2.0 functional API.

Note: the es_variants family names its frozen config dataclasses with a
``*Config`` suffix; this module re-exports them under the bare
algorithm names the public API exposes.
"""
from .de_variants import (
    DE,
    SHADE,
    CoDE,
    SaDE,
    ODE,
    JaDE,
    make_code,
    make_de,
    make_jade,
    make_ode,
    make_sade,
    make_shade,
)
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
    VirtualES,
    VirtualESConfig,
    VirtualLoRAES,
    VirtualLoRAESConfig,
    XNESConfig,
    make_open_es,
    make_xnes,
    make_separable_nes,
    make_des,
    make_snes,
    make_ars,
    make_asebo,
    make_persistent_es,
    make_noise_reuse_es,
    make_guided_es,
    make_esmc,
    make_cma_es,
    make_virtual_es,
    make_virtual_lora_es,
)
from .pso_variants import (
    CLPSO,
    CSO,
    DMSPSOEL,
    FSPSO,
    PSO,
    SLPSOGS,
    SLPSOUS,
    make_clpso,
    make_cso,
    make_dms_pso_el,
    make_fs_pso,
    make_pso,
    make_sl_pso_gs,
    make_sl_pso_us,
)

# 1.x-style bare-name aliases for the ES family
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
    "make_code",
    "make_de",
    "make_jade",
    "make_ode",
    "make_sade",
    "make_shade",
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
    "make_open_es",
    "make_xnes",
    "make_separable_nes",
    "make_des",
    "make_snes",
    "make_ars",
    "make_asebo",
    "make_persistent_es",
    "make_noise_reuse_es",
    "make_guided_es",
    "make_esmc",
    "make_cma_es",
    # ES Variants — virtual (training-free) family
    "VirtualESConfig",
    "VirtualLoRAESConfig",
    "VirtualES",
    "VirtualLoRAES",
    "make_virtual_es",
    "make_virtual_lora_es",
    # PSO Variants
    "CLPSO",
    "CSO",
    "DMSPSOEL",
    "FSPSO",
    "PSO",
    "SLPSOGS",
    "SLPSOUS",
    "make_clpso",
    "make_cso",
    "make_dms_pso_el",
    "make_fs_pso",
    "make_pso",
    "make_sl_pso_gs",
    "make_sl_pso_us",
]
