from .standard import StdPipeline
from .multidevice import MultiDevicePipeline
from .gym import GymPipeline
from .uni_workflow import UniWorkflow

try:
    # optional dependency: ray
    from .distributed import RayDistributedWorkflow
except ImportError as e:
    original_erorr_msg = str(e)

    def DistributedPipeline(*args, **kwargs):
        raise ImportError(
            f'DistributedPipeline requires ray, but got "{original_erorr_msg}" when importing'
        )
