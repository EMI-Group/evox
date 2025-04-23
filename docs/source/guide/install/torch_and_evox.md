# EvoX Installation Guide

## Install EvoX

EvoX is available at PyPI and can be installed via:

```bash
# install pytorch first
# for example:
pip install torch

# then install EvoX
pip install "evox[default]"
```

You can also assign extra options during the installation, currently available extras are `vis`, `neuroevolution`, `test`, `docs`, `default`. For example, to install EvoX with all features, run the following command:

```bash
pip install "evox[vis,neuroevolution]"
```

## Install PyTorch with accelerator support

`evox` relies on `torch` to provide hardware acceleration.
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

To summarize, whether `evox` has CPU support or Nvidia GPU support (CUDA) or AMD GPU support (ROCm) depends on the installed PyTorch version. Please refer to the PyTorch official website for more installation help: [`torch`](https://pytorch.org/)


## Nvidia GPU support on Windows

EvoX support GPU acceleration through PyTorch.
There are two ways to use PyTorch with GPU acceleration on Windows:

1. Using WSL 2 (Windows Subsystem for Linux) and install PyTorch on the Linux side.
2. Directly install PyTorch on Windows.

For option 2, we provide a [one-click script](/_static/win-install.bat) for fast deployment on fresh installed windows 10/11 64bit with Nvidia GPUs. The script will not use WSL 2 and will install the native Pytorch version on Windows. It will automatically install related applications like VSCode, Git and MiniForge3.

* Ensure the [Nvidia driver](https://www.nvidia.com/Download/index.aspx?lang=en-us) is properly installed first. Otherwise the script will fall back to cpu mode.
* When running the script, ensure a stable network (accessible to `github.com` etc.).
* If the script is failed due to network failure, close and reopen it to continue the installation.

### Manual installation on Windows

If you prefer to install PyTorch directly on Windows manually, you can follow the steps below:
1. Install Nvidia driver as mentioned above.
2. Install Python 3.10 or above from [python.org](https://www.python.org/downloads/).
3. Install PyTorch.
4. (Optional) Install [`triton-windows`](https://github.com/woct0rdho/triton-windows) for `torch.compile` support on Windows.
5. Install EvoX.

### Windows WSL 2

Download the [latest NVIDIA Windows GPU Driver](https://www.nvidia.com/Download/index.aspx?lang=en-us) and install it. Then your WSL 2 will support Nvidia GPUs in its Linux environments.

```{warning}
Do **NOT** install any NVIDIA GPU Linux driver within WSL 2. Install the driver on the Windows side.
```

```{seealso}
NVIDIA has a detailed [CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
```

## AMD GPU (ROCm) support

We recommend using a Docker container from [`rocm/pytorch`](https://hub.docker.com/r/rocm/pytorch).

```shell
docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 8G -v $HOME/dockerx:/dockerx -w /dockerx rocm/pytorch​:latest
```

## Verify the installation

Open a Python terminal, and run the following:

```python
from torch.utils.collect_env import get_pretty_env_info
import evox

print(get_pretty_env_info())
```
