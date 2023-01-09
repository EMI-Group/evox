from .torchvision_dataset import TorchvisionDataset

try:
    # optional dependency: torchvision, optax
    from .torchvision_dataset import TorchvisionDataset
except ImportError as e:
    original_erorr_msg = str(e)

    def TorchvisionDataset(*args, **kwargs):
        raise ImportError(
            f'TorchvisionDataset requires torchvision, optax but got "{original_erorr_msg}" when importing'
        )
