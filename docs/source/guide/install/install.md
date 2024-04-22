# EvoX Installation Guide

## Install EvoX

EvoX is available at Pypi and can be installed via:

```bash
pip install evox
```

To install EvoX with optional dependencies:

```bash
pip install evox[<feature1>,<feature2>]
```

available features are `gymnasium`, `neuroevolution`, `envpool`, `distributed`, and `full` which concludes all features.
For example, to install EvoX with all features, do:

```bash
pip install evox[full]
```

## Install JAX with accelerator support

`EvoX` relies on `JAX` to provide hardware acceleration.
The overall architecture of these Python packages looks like this:

```{mermaid}
stateDiagram-v2
    jaxlibCuda : jaxlib-cuda
    gpu : NVIDIA GPU

    direction LR

    evox --> jax
    jax --> jaxlib
    jax --> jaxlibCuda
    jaxlib --> CPU
    jaxlibCuda --> gpu
```

`JAX` itself is pure Python, and `jaxlib` provides the C/C++ code.
To utilize JAX's hardware acceleration ability, make sure to install the correct `jaxlib` version.

To summarize, you will need the follow 3 things to enable accelerator support:
1. GPU driver
2. CUDA libraries
3. The correct jaxlib version (the one with accelerator support).

```{seealso}
For more information, e.g. other platforms, please check out JAX's [official installation guide](https://github.com/google/jax/?tab=readme-ov-file#installation).
```

### CPU only

```bash
pip install -U "jax[cpu]"
```

### CUDA (NVIDIA GPU)

To enable CUDA acceleration, please ensure that the following components are installed in a compatible manner:

```{mermaid}
stateDiagram-v2
    jaxlib : jaxlib-cuda
    cuda : CUDA libraries
    driver : GPU driver
    gpu : NVIDIA GPU
    user: User Space
    kernel: Kernal Space

    direction LR

    state user {
      direction LR
      jaxlib --> cuda
    }

    cuda --> driver

    state kernel {
      direction LR
      driver --> gpu
    }
```

```{note}
If your using any virtualization technology, like WSL, docker.
- **kernel space components**: should be installed on your host system.
  For example, if you are using WSL with Windows, then the driver should be installed on Windows, not inside WSL.
  If you are using container (e.g. docker), then the driver should be installed on your host OS (outside docker).
- **user space components**: need to be installed inside WSL or docker.
```

#### Install NVIDIA's proprietary driver

Please notice that this step requires administrative privileges and a reboot.
So, if you are using a shared server, please contact the server's administrator for support.

##### Windows WSL 2

Download the [latest NVIDIA Windows GPU Driver](https://www.nvidia.com/Download/index.aspx?lang=en-us), and install it.

```{warning}
You must **NOT** install any NVIDIA GPU Linux driver within WSL 2.
GPU driver this a kernel space program, so it should be installed on the Windows side.
```

```{seealso}
NVIDIA has a detailed [CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
```

##### GNU/Linux

```{warning}
If you are on GNU/Linux,
I strongly recommend to install the driver via the package manager of your Linux distribution.
Please do **NOT** install the driver from NVIDIA's website.
```

```{note}
Only NVIDIA's proprietary driver works with CUDA, open-source drivers like Nouveau do not.
```

The detailed installation guide depends on your operating system, for example

- ArchLinux
  ```bash
  sudo pacman -S nvidia nvidia-utils
  ```
- Ubuntu 22.04
  ```bash
  sudo apt install nvidia-driver-510 nvidia-utils-510
  ```
- NixOS
  first enable unfree software, and then add the following to your configuration.
  ```
  services.xserver.videoDrivers = [ "nvidia" ];
  ```

After installing the driver, please reboot.


##### Cluster (e.g. slurm, k8s)

If the latest driver has already been installed in the cluster, please go ahead and skip this section.

Otherwise, please contact the administrator of the cluster to upgrade the GPU driver version.
It is important to note that the driver must be installed on the host system,
rendering any effort within the container (e.g. docker, singularity) meaningless.
Thus only the administrator can solve this problem.

#### Install CUDA libraries

CUDA libraries are user space libraries, so you don't need to reboot after installation.
Again, it depends on your operating system, for example

- ArchLinux
  ```bash
  sudo pacman -S cuda cudnn nccl
  ```
- Ubuntu 22.04
  ```bash
  sudo apt install nvidia-cuda-toolkit nvidia-cudnn
  ```


Now, you can check your do
```bash
nvidia-smi
```
to see if your GPU is recognized by the driver.
If you see something like this, then you are good to go.

```
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.86.05              Driver Version: 535.86.05    CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 3090        Off | 00000000:00:00.0 Off |                  N/A |
| 35%   35C    P8              25W / 350W |     27MiB / 24576MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
+---------------------------------------------------------------------------------------+
```

Then check your jaxlib version by

```bash
pip show jaxlib
```

Please make sure you jaxlib's CUDA version is smaller or equal to the CUDA version of this host.
```
Name: jaxlib
Version: 0.4.18+cuda11.cudnn86
Summary: XLA library for JAX
Home-page: https://github.com/google/jax
Author: JAX team
Author-email: jax-dev@google.com
License: Apache-2.0
Location: /----/python3.10/site-packages
Requires: ml-dtypes, numpy, scipy
```

For example, we have `0.4.18+cuda11.cudnn86` installed, and 11 < 12.2 (displayed by `nvidia-smi`). So we are good to go.

```{tip}
Since installing `jax[cuda12]` will usually install the jaxlib compiled with the latest CUDA version.
Even if you have CUDA 12, your CUDA version might still be lower than the version of that jaxlib requires.
In this case, try to install `jax[cuda11]`.
```

### AMD GPU (ROCM)

Despite being considered experimental, installing AMD GPUs for ROCm is surprisingly straightforward thanks to their open-source drivers. However, currently only a limited number of GPUs are supported, notably the Radeon RX 7900XTX and Radeon PRO W7900 for consumer-grade GPUs. Note that Windows is not currently supported.

#### Install GPU driver

Since the AMD driver is open-source, installation is simplified: simply install mesa through your Linux distribution's package manager. In many cases, the driver may already be pre-installed.

To verify that the driver is installed, run the following command:

```bash
lsmod | grep amdgpu
```

And you should see `amdgpu` in the output.

#### Install ROCm

The latest version of ROCm (v5.7.1 or later) may not be available in your Linux distribution's package manager. Therefore, using a containerized environment is the easiest way to get started.

```bash
docker run -it --network=host --device=/dev/kfd --device=/dev/dri --ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined rocm/jax:latest
```

Please visit [Docker Hub](https://hub.docker.com/r/rocm/jax) for further instructions.

## Verify your installation

Open a Python terminal, and run the following:

```python
import evox
import jax
jax.numpy.arange(10)
```

Here are some possible output:

````{tab-set}

  ```{tab-item} Correct

    ```python
    >>> import evox
    >>> import jax
    >>> jax.numpy.arange(10)
    Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int32)
    ```

  ```

  ```{tab-item} EvoX not installed

    ```python
    >>> import evox
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ModuleNotFoundError: No module named 'evox'
    ```

  ```

  ```{tab-item} Wrong jaxlib version

    ```python
    >>> import evox
    An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.
    >>> import jax
    >>> jax.numpy.arange(10)
    Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int32)
    ```

  ```

  ```{tab-item} Wrong driver/library

    ```python
    >>> import evox
    >>> import jax
    >>> jax.numpy.arange(10)
    Could not load library libcublasLt.so.11. Error: libcublasLt.so.11: cannot open shared object file: No such file or directory
    Aborted (core dumped)
    ```

  ```

````
