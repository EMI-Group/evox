# Windows with GPU acceleration

Currently the easiest way to install JAX with GPU acceleration on Windows is to use the `WSL 2`.

## Install WSL 2

Windows has a detailed guide on how to install WSL 2 [here](https://learn.microsoft.com/en-us/windows/wsl/install)

The simple way to install a Linux distribution is to use the `Windows Store` and search for the name of the distribution (e.g. ubuntu, debian) and click the install button.

## Install NVIDIA driver

```{warning}
You must **NOT** install any NVIDIA GPU Linux driver within WSL 2.
GPU driver this a kernel space program, so it should be installed on the Windows side.
```

Go to [here](https://www.nvidia.com/Download/index.aspx), to download the latest NVIDIA Windows GPU Driver, and install it.