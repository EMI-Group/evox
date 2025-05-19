# 2. 安装与环境配置

在开始使用 EvoX 之前，需要正确安装软件及其依赖环境。本章将分别介绍 Windows 和 Linux 下的安装步骤，以及必要的依赖项准备和配置方法。请确保在安装前满足基本的系统要求：**Python 3.10+**、足够的磁盘空间，以及（可选）支持的 GPU 和相应驱动。

## 依赖项和前置准备

- **Python 环境**：EvoX 基于 Python 开发，请确保系统已安装 Python 3.10 或更高版本。建议使用虚拟环境（如 `venv`）可以避免依赖冲突。

- **PyTorch**：EvoX 利用 PyTorch 提供底层的张量计算和硬件加速支持。因此，在安装 EvoX 前**必须先安装 PyTorch**。安装时请根据您的硬件选择对应版本：如果有 NVIDIA GPU，则安装包含 CUDA 支持的 PyTorch；如果只有 CPU 或使用 AMD GPU，则安装相应版本：CPU 版或 ROCm 版。可以参考 [PyTorch 官方指南](https://pytorch.org) 获取合适的安装命令，例如：

  ```bash
  # 例如，通过 pip 安装 CUDA (Nvidia) 版本的 PyTorch
  pip install torch torchvision torchaudio

  # 如果是AMD显卡，通过 pip 安装 ROCm (AMD) 版本的 PyTorch
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2.4

  # CPU版本：
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  ```

在进行正式安装前，建议更新 `pip` 到最新版本，并确保网络连接通畅（因为需要从 PyPI 下载包）。准备好上述环境后，即可开始安装 EvoX。

### Windows 下的安装步骤

Windows 用户可以选择**自动脚本安装**或**手动安装**。官方提供了一键安装脚本，可在全新环境下一键部署 EvoX 及所需依赖，但手动安装可以让您更了解每一步细节。下面我们分别介绍两种方式：

**方式一：使用一键安装脚本（win-install.bat）**
EvoX 提供了适用于 Windows 10/11 (64位) 的[快速安装脚本](/_static/win-install.bat)。脚本会自动安装 Miniforge3 (轻量版  Conda)、最新的 Python 和 PyTorch（含CUDA支持）、EvoX，以及实用工具如 VSCode 和 Git。使用方法：

1. 从官方获取 `win-install.bat` 脚本（可在 EvoX 文档或GitHub中找到下载链接）。确保您的Windows已安装[NVIDIA驱动](https://www.nvidia.com/en-us/drivers/)且网络良好。
2. 运行该脚本。该脚本运行本身不需要管理员权限，但是过程中系统可能弹出权限请求，允许运行即可。脚本将自动执行环境配置和安装，全程需要联网下载多个组件。
3. 等待脚本完成安装。安装成功后，会看到提示并可能自动打开 VSCode。此时 EvoX 及其依赖已安装完毕。

> **注意**：如果在执行脚本过程中网络中断或失败，请关闭脚本窗口并重新运行。脚本具有断点续传能力，多次运行会继续未完成的部分。

**方式二：手动安装**
若您希望自行控制每一步，可按照以下步骤手动安装：

1. **安装显卡驱动**：确保已安装最新的 NVIDIA 显卡驱动（通过[NVIDIA官网](https://www.nvidia.cn/Download/index.aspx)获取）。没有独立显卡的用户可跳过此步。

2. **安装 Python**：下载并安装 [Python 3.10+ Windows安装包](https://www.python.org/downloads/windows/)。安装时勾选“Add Python to PATH”以便后续使用 `pip`。

3. **安装 PyTorch**：打开命令提示符（CMD）或 PowerShell，根据硬件选择安装命令：

   - 参考 [PyTorch官网](https://pytorch.org/get-started/locally/) 获取适配您GPU或CPU版本的 pip 安装命令。例如：

     ```bash
     pip install torch torchvision torchaudio
     ```

4. **（可选）安装 Triton 编译器**：在 Windows 上，PyTorch 默认不包含 Triton 支持, 若您希望使用 PyTorch 2.0的`torch.compile`特性提升性能，可通过安装第三方(triton-windows)[https://github.com/woct0rdho/triton-windows]来获取支持。这一步不是必须的，但对性能优化章节有帮助。

5. **安装 EvoX**：在命令行执行：

  ```bash
  pip install "evox[default]"

  # 此外支持多种extra选项来预装其他组件
  pip install "evox[vis]" # visualization support
  pip install "evox[neuroevolution]" # NE support
  ```

  ````{note}
  某些软件包可能需要额外的系统依赖项。如果是这种情况，安装程序会提示类似以下的消息：

  ```console
  error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/
  ```

  遇到此类提示时，请按照提供的指引安装所需的系统依赖项后再继续操作。
  ````



### Linux 下的安装步骤

Linux 系统（如 Ubuntu）下安装 EvoX 相对直接，大部分情况下可以通过 `pip` 完成。以下是一般步骤：

1. **安装系统依赖**：确保已安装基本的开发工具和 Python。建议使用最新的 Python 3.10+版本，可通过系统包管理器（apt、yum等）或 Anaconda 来安装。

2. **安装 GPU 驱动**（如果使用GPU）：在 Linux 上，需安装 NVIDIA 驱动。Ubuntu 用户可以通过 apt 安装驱动。安装后，使用 `nvidia-smi` 命令确认驱动工作正常。如果没有 GPU 或使用 CPU ，可跳过此步。

```{note}
如果是 WSL , **不要**在其 Linux 子系统中安装 NVIDIA 驱动，应在 Windows 端安装。
```

```{tip}
你很可能只需要安装驱动程序，而**不需要**单独安装 CUDA 或其他依赖项。
这些库已经通过 pip 安装的 PyTorch 包含在内。
```

```{tip}
所需的驱动程序版本取决于你的硬件。如果你使用的是较新的 NVIDIA 显卡，通常推荐安装最新版驱动程序。
为了获得更好的兼容性以及使用最新驱动程序，通常建议使用较新的 Linux 发行版（例如 Ubuntu 25.04 而不是 22.04）。
```


3. **安装 PyTorch**：同 Windows 类似，先安装 PyTorch 以确保硬件加速正常。可以参考 [PyTorch 官方指南](https://pytorch.org)

4. **安装 EvoX**：运行

  ```bash
  pip install evox
  ```

  如果您计划使用可视化或神经进化等扩展功能，可以一次性安装带所有附加功能的版本，例如：

  ```bash
  pip install evox[vis,neuroevolution]
  ```

  这将同时安装可视化模块和 Brax 等神经进化相关依赖 ([EvoX Installation Guide](#EvoX Installation Guide))。您也可以根据需要选择 extras，比如只安装 `vis` 或 `neuroevolution`。


#### 使用容器安装 (Docker, Podman)

对于 AMD GPU 用户或希望隔离环境的用户，官方建议使用 Docker 镜像。例如，使用带 ROCm 的 PyTorch 官方Docker 镜像可以避免繁琐的环境配置。执行类似如下的命令运行容器：

```bash
docker run -it --gpus all --shm-size=8g rocm/pytorch:latest
```

然后在容器内安装 EvoX（同上面的pip步骤）。这种方式可以方便地获取GPU加速支持。



## 验证 EvoX 运行环境

完成安装后，您可以通过下面的步骤验证EvoX是否正常工作。

- **验证安装**：打开终端/命令提示符，进入Python解释器，执行以下代码检查环境信息：

  ```python
  from torch.utils.collect_env import get_pretty_env_info
  import evox
  print(get_pretty_env_info())
  ```

  该代码将打印出 PyTorch 和系统的环境配置，如果包含 EvoX 并且没有错误信息，说明安装成功。您也可以尝试 `import evox; print(evox.__version__)` 查看EvoX版本号，确认无误。

- **其他环境配置**：根据需要，您可以调整线程数等影响性能的参数。例如，设置环境变量 `OMP_NUM_THREADS` 控制CPU上并行线程数， 增加共享内存(`--shm-size`)避免 Docker 容器内的内存不足等。如果您使用 Jupyter Notebook 或 PyCharm 等 IDE 进行开发，请确保其 Interpreter 使用的是刚安装 EvoX 的 Python 环境。

完成以上配置，您的开发环境就搭建好了。接下来，我们将介绍如何在这个环境中开始使用 EvoX 进行优化任务。
