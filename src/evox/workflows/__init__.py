from .standard import StdWorkflow
from .uni_workflow import UniWorkflow

try:
    # optional dependency: ray
    from .distributed import RayDistributedWorkflow
except ImportError as e:
    original_error_msg = str(e)

    def RayDistributedWorkflow(*args, **kwargs):
        raise ImportError(
            f'RayDistributedWorkflow requires ray, but got "{original_error_msg}" when importing'
        )
