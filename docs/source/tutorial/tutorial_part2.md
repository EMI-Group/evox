# 2. Installation and Environment Setup

Before using EvoX, you need to properly install the software and its dependencies. This chapter covers installation steps for both Windows and Linux, as well as how to prepare and configure the required dependencies. Make sure you meet the basic system requirements before installation: **Python 3.10+**, sufficient disk space, and optionally a supported GPU with the appropriate driver.

## Dependencies and Preparations

- **Python Environment**: EvoX is built on Python, so ensure Python 3.10 or higher is installed. It’s recommended to use a virtual environment (such as `venv`) to avoid dependency conflicts.

- **PyTorch**: EvoX uses PyTorch for tensor operations and hardware acceleration. Therefore, **PyTorch must be installed before installing EvoX**. Choose the version based on your hardware: install the CUDA version if you have an NVIDIA GPU, the ROCm version for AMD GPUs, or the CPU version if no GPU is available. Refer to the [official PyTorch guide](https://pytorch.org) for the appropriate command, for example:

  ```bash
  # For NVIDIA GPUs (CUDA)
  pip install torch torchvision torchaudio

  # For AMD GPUs (ROCm)
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2.4

  # For CPU-only
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  ```

It’s recommended to update `pip` to the latest version and ensure a stable internet connection before installation (packages will be downloaded from PyPI). Once the environment is ready, you can install EvoX.

### Installation on Windows

Windows users can choose **automatic script installation** or **manual installation**. The official one-click installer provides an easy way to set up EvoX and its dependencies in a clean environment, but manual installation allows more control.

**Option 1: Using the One-Click Installation Script (win-install.bat)**
EvoX provides a [quick installation script](/_static/win-install.bat) for Windows 10/11 (64-bit). The script installs Miniforge3 (a lightweight Conda), Python, PyTorch (with CUDA), EvoX, and useful tools like VSCode and Git. To use:

1. Download `win-install.bat` from the EvoX docs or GitHub. Make sure you have a [NVIDIA driver](https://www.nvidia.com/en-us/drivers/) installed and a stable internet connection.
2. Run the script. It doesn’t require admin privileges, but may request permission during execution—allow it. The script will install and configure everything automatically.
3. Wait for completion. Upon success, you’ll see a message and possibly VSCode opening. EvoX and its dependencies will be installed.

> **Note**: If the script fails due to network issues, close it and rerun. It supports resume on failure.

**Option 2: Manual Installation**
To manually install EvoX:

1. **Install GPU Driver**: Install the latest NVIDIA driver from the [official website](https://www.nvidia.cn/Download/index.aspx). If no dedicated GPU, skip this step.

2. **Install Python**: Download [Python 3.10+ for Windows](https://www.python.org/downloads/windows/) and enable “Add Python to PATH” during installation.

3. **Install PyTorch**: Open CMD or PowerShell and install PyTorch based on your hardware:

   ```bash
   pip install torch torchvision torchaudio
   ```

4. **(Optional) Install Triton Compiler**: PyTorch on Windows lacks Triton support. If you want to use `torch.compile` (available in PyTorch 2.0), install the third-party [triton-windows](https://github.com/woct0rdho/triton-windows). Optional but useful for performance optimization.

5. **Install EvoX**:

   ```bash
   pip install "evox[default]"

   # Optional extras:
   pip install "evox[vis]"           # Visualization support
   pip install "evox[neuroevolution]" # Neuroevolution support
   ```


  ````{note}
  Some packages may require additional system dependencies. If this is the case, the installer will prompt you with a message like the following:

  ```console
  error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/
  ```

  When you encounter such messages, follow the provided instructions to install the necessary dependencies before proceeding.
  ````


### Installation on Linux

Installing EvoX on Linux (e.g., Ubuntu) is straightforward and mostly handled via `pip`.

1. **Install System Dependencies**: Make sure basic development tools and Python 3.10+ are installed. You can use a package manager (apt, yum) or Anaconda.

2. **Install GPU Driver** (if using GPU): Use the appropriate package manager (e.g., `apt`) to install NVIDIA drivers. Verify installation with `nvidia-smi`. Skip if using CPU.

```{note}
On WSL, **do not** install NVIDIA drivers inside the Linux subsystem—install them on the Windows side.
```

```{tip}
It's very likely that you only need to install the driver, but do NOT need to install CUDA or other dependencies.
Those libraries are already included in the PyTorch installation via pip.
```

```{tip}
The required driver version depends on your hardware. If you have a recent NVIDIA GPU, using the latest driver version is often the best choice.
To ensure better compatibility and access to the latest drivers, it's generally a good idea to use a newer Linux distribution (e.g., Ubuntu 25.04 instead of 22.04).
```

1. **Install PyTorch**: As on Windows, install based on hardware. Refer to the [PyTorch official guide](https://pytorch.org).

2. **Install EvoX**:

   ```bash
   pip install evox
   ```

   Or with extras:

   ```bash
   pip install evox[vis,neuroevolution]
   ```

   This installs visualization modules and neuroevolution dependencies (like Brax). You can also choose individual extras like `vis` or `neuroevolution`.

#### Container Installation (Docker, Podman)

For AMD GPU users or those seeking environment isolation, Docker is recommended. For example, using the official PyTorch Docker image with ROCm:

```bash
docker run -it --gpus all --shm-size=8g rocm/pytorch:latest
```

Inside the container, install EvoX as usual using `pip`.

## Verifying EvoX Installation

To verify that EvoX is properly installed:

- **Basic Check**: In terminal or Python shell, run:

  ```python
  from torch.utils.collect_env import get_pretty_env_info
  import evox
  print(get_pretty_env_info())
  ```

  This prints PyTorch and system configuration info. If EvoX is imported without errors, the installation was successful. You can also check the version:

  ```python
  import evox
  print(evox.__version__)
  ```

- **Optional Settings**: You may tune performance-related settings, such as:

  - Setting environment variables like `OMP_NUM_THREADS` to control CPU thread count
  - Increasing Docker shared memory with `--shm-size`
  - Ensuring your IDE (Jupyter, PyCharm, etc.) uses the correct Python environment

Once the setup is complete, you're ready to start optimizing with EvoX.
