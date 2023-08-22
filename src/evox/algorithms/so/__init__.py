from .amalgam import AMaLGaM, IndependentAMaLGaM
from .ars import ARS
from .asebo import ASEBO
from .cma_es import BIPOPCMAES, CMAES, IPOPCMAES, SepCMAES
from .cr_fm_nes import CR_FM_NES
from .cso import CSO
from .de import DE
from .des import DES
from .esmc import ESMC
from .guided_es import GuidedES
from .ma_es import LMMAES, MAES
from .nes import SeparableNES, xNES
from .noise_reuse_es import Noise_reuse_es
from .open_es import OpenES
from .persistent_es import PersistentES
from .pgpe import PGPE
from .pso_varients import *
from .rmes import RMES
from .snes import SNES
from .code import CoDE
from .jade import JaDE
from .sade import SaDE
from .shade import SHADE

try:
    # optional dependency: flax
    from .les import LES
except ImportError as e:
    original_erorr_msg = str(e)

    def LES(*args, **kwargs):
        raise ImportError(
            f'LES requires flax but got "{original_erorr_msg}" when importing'
        )
