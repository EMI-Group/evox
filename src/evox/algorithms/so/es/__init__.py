from .pgpe import PGPE
from .nes import xNES, SeparableNES
from .open_es import OpenES
from .ars import ARS
from .amalgam import AMaLGaM, IndependentAMaLGaM
from .ma_es import MAES, LMMAES
from .rmes import RMES
from .asebo import ASEBO
from .noise_reuse_es import Noise_reuse_es
from .snes import SNES
from .cr_fm_nes import CR_FM_NES
from .des import DES
from .esmc import ESMC
from .guided_es import GuidedES
from .persistent_es import PersistentES


try:
    # optional dependency: flax
    from .les import LES
except ImportError as e:
    original_erorr_msg = str(e)

    def LES(*args, **kwargs):
        raise ImportError(
            f'LES requires flax but got "{original_erorr_msg}" when importing'
        )