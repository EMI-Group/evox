from .standard import StdPipeline
from .gym import GymPipeline

try:
    # optional dependency: ray
    from .distributed import DistributedPipeline
except ImportError as e:
    original_erorr_msg = str(e)

    def DistributedPipeline(*args, **kwargs):
        raise ImportError(
            f'DistributedPipeline requires ray, but got "{original_erorr_msg}" when importing'
        )
