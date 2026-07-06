import functools
import importlib

import torch

_triton_checked = False
_triton_available = False


def _check_triton():
    """Lazily check if Triton is importable. Caches the result."""
    global _triton_checked, _triton_available
    if not _triton_checked:
        _triton_available = importlib.util.find_spec("triton") is not None
        _triton_checked = True
    return _triton_available


def has_triton():
    """Check whether the Triton package is importable.

    :return: True if Triton can be imported, False otherwise.
    """
    return _check_triton()


def triton_supports_device(device: torch.device | str) -> bool:
    """Check whether Triton can execute kernels on the given device.

    Triton kernels currently only run on CUDA devices. This function returns
    True only when Triton is importable **and** the device is CUDA **and**
    CUDA is available on the host.

    :param device: A torch.device or device string (e.g. "cuda", "cpu", "mps").
    :return: True if a Triton kernel can run on this device.
    """
    if not has_triton():
        return False
    device = torch.device(device) if isinstance(device, str) else device
    return device.type == "cuda" and torch.cuda.is_available()
