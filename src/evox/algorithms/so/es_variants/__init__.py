from .amalgam import AMaLGaM, IndependentAMaLGaM
from .ars import ARS
from .asebo import ASEBO
from .cma_es import CMAES, SepCMAES, IPOPCMAES, BIPOPCMAES
from .cr_fm_nes import CR_FM_NES
from .des import DES
from .esmc import ESMC
from .guided_es import GuidedES
from .ma_es import MAES, LMMAES
from .nes import XNES, SeparableNES
from .noise_reuse_es import Noise_reuse_es
from .open_es import OpenES
from .persistent_es import PersistentES
from .pgpe import PGPE
from .rmes import RMES
from .snes import SNES

try:
    # optional dependency: flax
    from .les import LES
except ImportError as e:
    original_error_msg = str(e)

    def LES(*args, **kwargs):
        raise ImportError(
            f'LES requires flax but got "{original_error_msg}" when importing'
        )
