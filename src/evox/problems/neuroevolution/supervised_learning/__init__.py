try:
    # optional dependency: torchvision, optax
    from .tfds import TensorflowDataset
except ImportError as e:
    original_error_msg = str(e)

    def TensorflowDataset(*args, **kwargs):
        raise ImportError(
            f'TensorflowDataset requires tensorflow-datasets, grain but got "{original_error_msg}" when importing'
        )
