# Windows with GPU acceleration

EvoX support GPU acceleration through `PyTorch`.
There are two ways to use PyTorch with GPU acceleration on Windows:
1. Using WSL 2 (Windows Subsystem for Linux) and install PyTorch on the Linux side.
2. Directly install PyTorch on Windows, but you won't have `jit` support.

## WSL 2

### Install WSL 2

Windows has a detailed guide on how to install WSL 2 [here](https://learn.microsoft.com/en-us/windows/wsl/install)

The simple way to install a Linux distribution is to use the `Windows Store` and search for the name of the distribution (e.g. ubuntu, debian) and click the install button.

### Install NVIDIA driver

```{warning}
You must **NOT** install any NVIDIA GPU Linux driver within WSL 2.
GPU driver this a kernel space program, so it should be installed on the Windows side.
```

Go to [here](https://www.nvidia.com/Download/index.aspx), to download the latest NVIDIA Windows GPU Driver, and install it.

## Directly install PyTorch on Windows

### Install PyTorch

You can install PyTorch on Windows by following the instructions on the [PyTorch website](https://pytorch.org/get-started/locally/).

```{note}
You won't have `jit` support if you install PyTorch on Windows.
```
