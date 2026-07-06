import importlib.util

import torch

_triton_checked = False
_triton_available = False

#: Set of device type strings for which Triton kernels are registered.
#: Defaults to ``{"cuda"}``; extend it with :func:`register_triton_device_type`
#: to support additional backends such as Ascend NPU (``"npu"``).
_triton_device_types: set[str] = {"cuda"}


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


def triton_device_types() -> frozenset[str]:
    """Return the set of device type strings for which Triton kernels are registered.

    Defaults to ``{"cuda"}``; may be extended via
    :func:`register_triton_device_type`.

    :return: A frozenset of registered device type strings.
    """
    return frozenset(_triton_device_types)


def register_triton_device_type(device_type: str) -> None:
    """Register an additional device type for Triton kernel support.

    This allows extending Triton coverage beyond the default ``"cuda"`` backend,
    e.g. to support Ascend NPU (``"npu"``). The input is normalized by stripping
    surrounding whitespace and lowercasing. Empty strings are ignored.

    :param device_type: The device type string to register (e.g. ``"npu"``).
    """
    normalized = device_type.strip().lower()
    if normalized:
        _triton_device_types.add(normalized)


def triton_supports_device(device: torch.device | str) -> bool:
    """Check whether Triton can execute kernels on the given device.

    Triton kernels run on CUDA devices (including AMD ROCm, which is aliased
    under the ``"cuda"`` device type) and, when registered, additional backends
    such as Ascend NPU (``"npu"``). This function returns True only when Triton
    is importable **and** the device type is among the registered Triton device
    types. For ``"cuda"`` devices, CUDA must additionally be available on the
    host; for other backends membership in the registered set is sufficient.

    :param device: A torch.device or device string (e.g. "cuda", "cpu", "npu").
    :return: True if a Triton kernel can run on this device.
    """
    if not has_triton():
        return False
    # Extract the device type without forcing a ``torch.device()`` conversion,
    # so that unrecognized device types (e.g. ``"npu"`` before the Ascend
    # backend is loaded) do not raise a ``RuntimeError``.
    if isinstance(device, str):
        device_type = device.split(":", 1)[0].strip()
    else:
        device_type = device.type
    if device_type not in _triton_device_types:
        return False
    if device_type == "cuda":
        return torch.cuda.is_available()
    return True
