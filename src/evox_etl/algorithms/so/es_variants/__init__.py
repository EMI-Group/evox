"""Functional (evox_etl) ports of the torch evox Evolution-Strategies family.

Mirrors the torch `__init__.py` exports (config dataclasses instead of
classes), plus a `make_*` functional constructor alongside each config.
Every module exposes the step protocol: plain `init(config, key) -> state`
and `step(config, state, evaluate) -> state` functions owning ONE full
generation each (sample → evaluate → update), with optional
`init_step`/`final_step` variants the workflow dispatches host-side.
The virtual (training-free) family is ported too: `virtual_es` stores only a
`(dim,)` center + `(pop_size,)` seeds and regenerates the full-parameter noise
on demand, `virtual_lora_es` the same with low-rank `B @ A` perturbations for
>= 2-D blocks (both use the shared `virtual_noise` generator and the
`(center, seeds, sigma)` evaluate payload).
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
    "VirtualESConfig",
    "make_virtual_es",
    "VirtualLoRAESConfig",
    "make_virtual_lora_es",
    # torch-style bare names for the virtual family (see the note below)
    "VirtualES",
    "VirtualLoRAES",
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
from .virtual_es import VirtualES, VirtualESConfig, VirtualLoRAES, make_virtual_es
from .virtual_lora_es import VirtualLoRAESConfig, make_virtual_lora_es

# Torch parity for the virtual family: torch `so/es_variants/__init__.py` exports
# the BARE names `VirtualES` / `VirtualLoRAES`, both coming from `virtual_es.py`
# (where `VirtualLoRAES = VirtualES`), so torch's bare `VirtualLoRAES` is the
# VirtualES class and the DISTINCT low-rank class in `virtual_lora_es.py` is
# shadowed (reachable only by its module path).  The same names are re-exported
# here; the distinct low-rank config is `VirtualLoRAESConfig` /
# `virtual_lora_es.VirtualLoRAES`.  The other families here keep the
# `*Config` + `make_*` convention.
