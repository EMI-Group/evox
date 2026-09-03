"""Functional (evox_etl) ports of the torch evox Evolution-Strategies family.

Mirrors the torch `__init__.py` exports (config dataclasses instead of
classes). VirtualLoRAES is intentionally NOT ported: it depends on torch-only
machinery (the triton_kernels Philox counter PRNG, LoRA factor/gradient
utilities) and a `(center, seeds, sigma)` tuple evaluate protocol that
evox_etl's tensor-only init/ask/tell contract does not support. See
`src/evox_etl/algorithms/CONTEXT.md` for the precise reasons.
"""

__all__ = [
    "OpenESConfig",
    "XNESConfig",
    "SeparableNESConfig",
    "DESConfig",
    "SNESConfig",
    "ARSConfig",
    "ASEBOConfig",
    "PersistentESConfig",
    "NoiseReuseESConfig",
    "GuidedESConfig",
    "ESMCConfig",
    "CMAESConfig",
]


from .ars import ARSConfig
from .asebo import ASEBOConfig
from .cma_es import CMAESConfig
from .des import DESConfig
from .esmc import ESMCConfig
from .guided_es import GuidedESConfig
from .nes import XNESConfig, SeparableNESConfig
from .noise_reuse_es import NoiseReuseESConfig
from .open_es import OpenESConfig
from .persistent_es import PersistentESConfig
from .snes import SNESConfig
