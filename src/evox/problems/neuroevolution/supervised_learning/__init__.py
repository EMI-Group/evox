try:
    # optional dependency: torchvision, optax
    from .torchvision_dataset import TorchvisionDataset
except ImportError as e:
    original_error_msg = str(e)

    def TorchvisionDataset(*args, **kwargs):
        raise ImportError(
            f'TorchvisionDataset requires torchvision, optax but got "{original_error_msg}" when importing'
        )
