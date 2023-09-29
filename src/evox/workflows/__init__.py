from .standard import StdWorkflow
from .uni_workflow import UniWorkflow

try:
    # optional dependency: ray
    from .distributed import RayDistributedWorkflow
except ImportError as e:
    original_erorr_msg = str(e)

    def DistributedWorkflow(*args, **kwargs):
        raise ImportError(
            f'DistributedWorkflow requires ray, but got "{original_erorr_msg}" when importing'
        )
