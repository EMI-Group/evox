"""Functional (evox_etl) ports of the torch evox Evolution-Strategies family.

Mirrors the torch `__init__.py` exports (config dataclasses instead of
classes), plus a `make_*` functional constructor alongside each config.
VirtualLoRAES is intentionally NOT ported: it depends on torch-only
machinery (the triton_kernels Philox counter PRNG, LoRA factor/gradient
utilities) and a `(center, seeds, sigma)` tuple evaluate protocol that
evox_etl's tensor-only init/ask/tell contract does not support. See
`src/evox_etl/algorithms/CONTEXT.md` for the precise reasons.
"""

__all__ = [
    "OpenESConfig",
    "make_open_es",
    "XNESConfig",
    "make_xnes",
    "SeparableNESConfig",
    "make_separable_nes",
    "DESConfig",
    "make_des",
    "SNESConfig",
    "make_snes",
    "ARSConfig",
    "make_ars",
    "ASEBOConfig",
    "make_asebo",
    "PersistentESConfig",
    "make_persistent_es",
    "NoiseReuseESConfig",
    "make_noise_reuse_es",
    "GuidedESConfig",
    "make_guided_es",
    "ESMCConfig",
    "make_esmc",
    "CMAESConfig",
    "make_cma_es",
]


from .ars import ARSConfig, make_ars
from .asebo import ASEBOConfig, make_asebo
from .cma_es import CMAESConfig, make_cma_es
from .des import DESConfig, make_des
from .esmc import ESMCConfig, make_esmc
from .guided_es import GuidedESConfig, make_guided_es
from .nes import XNESConfig, make_xnes, SeparableNESConfig, make_separable_nes
from .noise_reuse_es import NoiseReuseESConfig, make_noise_reuse_es
from .open_es import OpenESConfig, make_open_es
from .persistent_es import PersistentESConfig, make_persistent_es
from .snes import SNESConfig, make_snes
