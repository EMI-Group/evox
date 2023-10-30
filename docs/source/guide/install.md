# Installation Guide

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

    evox --> jax
    jax --> jaxlib
    jax --> jaxlibCuda
    jaxlib --> CPU
    jaxlibCuda --> gpu
```

`JAX` itself is pure Python, and `jaxlib` provides the C/C++ code.
To utilize JAX's hardware acceleration ability, make sure to install the correct `jaxlib` version.

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
    driver : CUDA driver
    gpu : NVIDIA GPU

    jaxlib --> cuda
    cuda --> driver
    driver --> gpu
```

#### Install NVIDIA's proprietary driver


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

```{note}
Only NVIDIA's proprietary driver works with CUDA, open-source drivers like Nouveau do not work.
```

```{warning}
If you are on GNU/Linux,
I strongly recommend to install the driver via the package manager of your Linux distribution.
Please do **NOT** install the driver from NVIDIA's website.
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

Please make sure you CUDA version is smaller or equal to the version of `jaxlib-cuda`.
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
