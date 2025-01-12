# EvoX Installation Guide

## Install EvoX

EvoX is available at pypi and can be installed via:

```bash
# install pytorch first
# for example:
pip install torch

# then install evox
pip install evox
```

## Install PyTorch with accelerator support

`EvoX` relies on `torch` to provide hardware acceleration.
The overall architecture of these Python packages looks like this:

```{mermaid}
stateDiagram-v2
    torch : torch
    nv_gpu : NVIDIA GPU
    amd_gpu : AMD GPU
    cpu : CPU

    direction LR

    evox --> torch
    torch --> nv_gpu
    torch --> amd_gpu
    torch --> cpu
```

To summarize, whether `EvoX` has CPU support or GPU support depends on the PyTorch version you installed. Please refere to the PyTorch official website for more installation help: [`torch`](https://pytorch.org/)


## Windows with GPU acceleration

EvoX support GPU acceleration through `PyTorch`.
There are two ways to use PyTorch with GPU acceleration on Windows:

1. Using WSL 2 (Windows Subsystem for Linux) and install PyTorch on the Linux side.
2. Directly install PyTorch on Windows, but you won't have `jit` support.

We also provide a [one-click script]() for windows 10/11 64bit with nvidia GPUs. The script will not use WSL 2 and will install the native Pytorch version on Windows.

### Windows WSL 2 (Advanced)

Download the [latest NVIDIA Windows GPU Driver](https://www.nvidia.com/Download/index.aspx?lang=en-us), and install it. Then your WSL 2 will support Nvidia GPUs in its Linux environments.

```{warning}
You must **NOT** install any NVIDIA GPU Linux driver within WSL 2.
GPU driver this a kernel space program, so it should be installed on the Windows side.
```

```{seealso}
NVIDIA has a detailed [CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
```

## Verify your installation

Open a Python terminal, and run the following:

```python
from torch.utils.collect_env import get_pretty_env_info
import evox

print(get_pretty_env_info())
```
